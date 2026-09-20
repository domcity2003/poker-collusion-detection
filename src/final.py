"""Final configuration: fit, score, and export a submission.

Selected on development folds 1-4. Three changes over the v2 submission
(public 0.83492), in descending order of how much I trust them:

1. BEHAVIOUR NOVELTY RULE. Evaluation contains an `other_coordination` family
   with zero development examples. A pair of that family can never earn
   behaviour credit, but guessing one of the three known families for it is not
   neutral - it inserts a high-risk false positive into that family's average
   precision. Labelling it `other_coordination` scores strictly better.
   Detection: a pair that the generic model calls colluding but that NO family
   specialist claims. Rehearsed by hiding each known family in turn (its truth
   label and its specialist) and scoring with the official metric: +0.010 to
   +0.017 on total score. This is the only change validated against the real
   scorer rather than a proxy.

2. FAMILY-SPECIALIST RISK BLEND. One detector per family, blended .65 generic
   /.35 max-specialist. The generic model lifts directed_transfer from AP .19
   to .51 but coordinated_isolation only .30 to .43, because isolation is 79 of
   304 training positives and a single model under-fits the minority signature.
   Generic is retained, not replaced: it is the only thing that can detect the
   unseen fourth family. Gain holds at every censoring level (see below).

3. Seed averaging over three seeds. No measured gain, kept as variance
   insurance on a submission that cannot be revised.

ON THE LOCAL METRIC. Broad AP (all unlabelled pairs treated as negatives) reads
.72 while the leaderboard implies real pair AP ~.905. The gap is undisclosed
positives: development labels only 1,860 of 137,739 candidates, whereas the
evaluation solution labels every colluding pair. Censoring the ~100 most
suspicious unknowns moves local AP to .94 and brackets reality. Judge changes at
several censoring levels, not at zero - see reports/metric_calibration.json.
"""
import sys,json,argparse,time
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
from catboost import CatBoostClassifier,CatBoostRanker,Pool
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import pair_table,feature_columns,hand_table,pool_blend,slim,FAM,ID
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import score as official_score,_average_precision

OUT=Path('artifacts/final');OUT.mkdir(parents=True,exist_ok=True)
SEEDS=(2026,7,991)
SPEC_BLEND=.35          # weight on max-specialist vs generic
NOVELTY_THR=.10         # below this, no family claims the pair
UNK_W=.15               # weight on the 136k unlabelled pairs

def mat(df,cols):return np.nan_to_num(df.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
def cb(seed,it=550,depth=5):
 return CatBoostClassifier(iterations=it,depth=depth,learning_rate=.055,l2_leaf_reg=8,
   thread_count=7,verbose=False,random_seed=seed,allow_writing_files=False)
def znorm(x,g):
 o=np.empty_like(x)
 for i in g.values():
  b=x[i];m=b.mean(0);s=b.std(0);s[s<1e-6]=1;o[i]=(b-m)/s
 return o
def grouping(p):
 g={}
 for i,v in enumerate(p):g.setdefault(v,[]).append(i)
 return {k:np.array(v) for k,v in g.items()}

def fit_pairs(df,cols,train):
 """Memory matters here: X is ~500MB, so every boolean-indexed copy counts.
 Each training subset is materialised ONCE and reused across seeds, and freed
 before the next family. Building it inside the seed loop instead is enough to
 exhaust an 8GB machine and have the process killed with no traceback."""
 import gc
 X=mat(df,cols);lab=df['label'].to_numpy();fam=df['behavior_family'].fill_null('none').to_numpy()
 y=(lab==1).astype(int);w=np.where(lab==1,8.,np.where(lab==0,1.,UNK_W))
 Xt,yt,wt=X[train],y[train],w[train]
 gens=[cb(s).fit(Xt,yt,sample_weight=wt) for s in SEEDS]
 del Xt,yt,wt;gc.collect()
 specs={}
 for fm in FAM:
  # Other families' positives are excluded, not relabelled negative: they are
  # colluding pairs, just not of this family.
  keep=train&~((lab==1)&(fam!=fm));yy=((lab==1)&(fam==fm)).astype(int)
  Xk,yk,wk=X[keep],yy[keep],w[keep]
  specs[fm]=[cb(s).fit(Xk,yk,sample_weight=wk) for s in SEEDS]
  del Xk,yk,wk;gc.collect()
  print('  fitted specialist',fm,flush=True)
 del X;gc.collect()
 return gens,specs

def predict_pairs(gens,specs,df,cols):
 X=mat(df,cols)
 gen=np.mean([m.predict_proba(X)[:,1] for m in gens],0)
 spec=np.vstack([np.mean([m.predict_proba(X)[:,1] for m in specs[fm]],0) for fm in FAM])
 del X
 risk=pool_blend((1-SPEC_BLEND)*gen+SPEC_BLEND*spec.max(0),df['pool'].to_numpy())
 claimed=spec.max(0)
 behavior=np.where(claimed<NOVELTY_THR,'other_coordination',np.array(FAM)[spec.argmax(0)])
 famprob=spec/np.maximum(spec.sum(0),1e-9)
 return np.clip(risk,0,1),behavior,famprob,gen,spec

def fit_evidence(train_mask=None):
 h=pl.read_parquet('artifacts/dev_hands_hindsight.parquet').filter(pl.col('label')==1).sort('pair_id')
 if train_mask is not None:h=h.filter(pl.Series(train_mask(h)))
 fc=[c for c,t in h.schema.items() if c not in ID and t.is_numeric()]
 X=mat(h,fc);y=h['ev'].to_numpy();pair=h['pair_id'].to_numpy();fam=h['behavior_family'].to_numpy()
 XX=np.hstack([X,znorm(X,grouping(pair))]);gid=np.searchsorted(np.unique(pair),pair)
 ms=[]
 for fm in FAM:
  s=fam==fm;o=np.argsort(gid[s],kind='stable');i=np.flatnonzero(s)[o]
  m=CatBoostRanker(loss_function='YetiRank',iterations=600,depth=5,learning_rate=.06,
    thread_count=7,verbose=False,random_seed=2026,allow_writing_files=False)
  m.fit(Pool(XX[i],y[i],group_id=gid[i]));ms.append(m)
 return ms,fc

def rank_hands(models,fc,h,probs):
 X=mat(h,fc);pair=h['pair_id'].to_numpy();g=grouping(pair)
 XX=np.hstack([X,znorm(X,g)])
 spec=np.vstack([m.predict(XX) for m in models]);hid=h['hand_id'].to_numpy();out={}
 for p,idx in g.items():
  w=probs.get(p,np.ones(3)/3);b=spec[:,idx]
  r=np.vstack([(np.argsort(np.argsort(x))+1)/len(x) for x in b])
  out[p]=list(hid[idx][np.argsort(-(w[:,None]*r).sum(0))[:5]])
 return out

def build_submission(df,cols,gens,specs,models,fc,phase):
 risk,behavior,famprob,gen,spec=predict_pairs(gens,specs,df,cols)
 probs={p:famprob[:,i] for i,p in enumerate(df['pair_id'])}
 top={}
 pairs=df.select('pair_id','player_1','player_2','pool')
 for i,pool in enumerate(sorted(df['pool'].unique().to_list())):
  pp=pairs.filter(pl.col('pool')==pool)
  if pp.height==0:continue
  top.update(rank_hands(models,fc,hand_table(phase,pool,pp),probs))
  if i%100==0:print('  evidence pool',i,flush=True)
 sub=pd.DataFrame({'pair_id':df['pair_id'].to_list(),'risk_score':risk,'predicted_behavior':behavior})
 for k in range(5):
  sub[f'evidence_hand_{k+1}']=[top.get(p,[])[k] if len(top.get(p,[]))>k else 'NO_EVIDENCE' for p in sub.pair_id]
 return sub,risk,spec

def censored_ap(y,s,lab,ks=(0,50,100,200,400)):
 out={}
 for k in ks:
  yy,ss=y,s
  if k:
   unk=np.flatnonzero(lab<0);drop=unk[np.argsort(-s[unk])[:k]]
   m=np.ones(len(s),bool);m[drop]=False;yy,ss=y[m],s[m]
  out[f'censor_{k}']=_average_precision(yy,ss)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--lockbox',action='store_true');args=ap.parse_args()
 t=time.time()
 # 250 features, chosen by importance on training folds only. Fewer features
 # scored BETTER than all 947 at every censoring level (.9525 vs .9483 at
 # censor_100) - 947 features against 372 positives is far past what the label
 # count supports - and it keeps the matrix at 138MB instead of 522MB, which is
 # the difference between fitting in 30s and thrashing an 8GB machine for
 # minutes per model.
 selm=json.loads(Path('artifacts/selected_features.json').read_text())
 dev=slim(pair_table('development'),selm['top250']);cols=feature_columns(dev)
 print('development table',dev.shape,'features',len(cols),flush=True)
 (OUT/'feature_columns.json').write_text(json.dumps(cols,indent=2))
 fold=dev['fold'].to_numpy();lab=dev['label'].to_numpy()
 if args.lockbox:
  train=fold!=0
  gens,specs=fit_pairs(dev,cols,train)
  models,fc=fit_evidence(lambda h:h['fold'].to_numpy()!=0)
  test=dev.filter(pl.Series(fold==0))
  y=(test['label'].to_numpy()==1).astype(int);tl=test['label'].to_numpy()
  labels=pl.read_parquet('artifacts/labels.parquet').filter(pl.col('fold')==0)
  evid=pl.read_csv('data/development_evidence.csv')
  risk,behavior,famprob,_,_=predict_pairs(gens,specs,test,cols)
  # The development per-hand files are keyed by the LABEL pair_id, while the
  # candidate table uses a synthetic 'player_1|player_2' id. Joining those two
  # on pair_id silently returns nothing and every pair ends up NO_EVIDENCE, so
  # the evidence stage is driven off the labels table here. Evaluation is not
  # affected: there both sides already use the official pair_id.
  lk=test.select('player_1','player_2','pair_id').join(
      labels.select('player_1','player_2',pl.col('pair_id').alias('lab_id')),on=['player_1','player_2'])
  m={r['pair_id']:r['lab_id'] for r in lk.iter_rows(named=True)}
  probs={m[p]:famprob[:,i] for i,p in enumerate(test['pair_id']) if p in m}
  risk_by={m[p]:risk[i] for i,p in enumerate(test['pair_id']) if p in m}
  beh_by={m[p]:behavior[i] for i,p in enumerate(test['pair_id']) if p in m}
  top={}
  for pool in sorted(labels['pool'].unique().to_list()):
   pp=labels.filter(pl.col('pool')==pool).select('pair_id','player_1','player_2')
   if pp.height==0:continue
   top.update(rank_hands(models,fc,hand_table('development',pool,pp),probs))
  assert len(top)>0,'evidence join produced nothing - check pair_id conventions'
  sol=labels.select('pair_id',pl.col('label').alias('risk_score'),
                    pl.col('behavior_family').alias('predicted_behavior')).to_pandas()
  for k in range(1,6):
   d=evid.filter(pl.col('evidence_rank')==k).select('pair_id',pl.col('hand_id').alias(f'evidence_hand_{k}')).to_pandas()
   sol=sol.merge(d,on='pair_id',how='left')
  sol=sol.fillna('NO_EVIDENCE')
  cur=pd.DataFrame({'pair_id':labels['pair_id'].to_list()})
  cur['risk_score']=[float(np.clip(risk_by.get(p,0),0,1)) for p in cur.pair_id]
  cur['predicted_behavior']=[beh_by.get(p,FAM[0]) for p in cur.pair_id]
  for k in range(5):
   cur[f'evidence_hand_{k+1}']=[top.get(p,[])[k] if len(top.get(p,[]))>k else 'NO_EVIDENCE' for p in cur.pair_id]
  evcols=[f'evidence_hand_{k}' for k in range(1,6)]
  n_ev=float((cur[evcols]!='NO_EVIDENCE').all(axis=1).mean())
  cur=cur.set_index('pair_id').loc[sol.pair_id].reset_index()
  # Behaviour-only ablation: what the novelty rule is actually worth here.
  plain=cur.copy();plain['predicted_behavior']=[
    FAM[int(np.argmax(probs[p]))] if p in probs else FAM[0] for p in plain.pair_id]
  res={'official_labeled_only':official_score(sol,cur,'pair_id'),
       'official_without_novelty_rule':official_score(sol,plain,'pair_id'),
       'pairs_with_full_evidence':n_ev,
       **censored_ap(y,risk,tl),
       'n_candidates':int(test.height),'n_positives':int(y.sum()),
       'flagged_other_coordination_all':float((behavior=='other_coordination').mean()),
       'flagged_other_coordination_positives':float((behavior[tl==1]=='other_coordination').mean()),
       'seconds':round(time.time()-t)}
  Path('reports/lockbox3.json').write_text(json.dumps(res,indent=2))
  print(json.dumps(res,indent=2),flush=True);return
 gens,specs=fit_pairs(dev,cols,np.ones(len(fold),bool))
 models,fc=fit_evidence()
 ev=slim(pair_table('evaluation'),selm['top250'])
 missing=[c for c in cols if c not in ev.columns];assert not missing,missing
 sub,risk,spec=build_submission(ev,cols,gens,specs,models,fc,'evaluation')
 sub.to_csv(OUT/'submission.csv',index=False)
 print('rows',len(sub),'flagged other_coordination',
       f'{(sub.predicted_behavior=="other_coordination").mean():.3%}',
       'seconds',round(time.time()-t),flush=True)
if __name__=='__main__':main()
