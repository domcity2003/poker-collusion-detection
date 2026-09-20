"""Evidence retrieval experiments, scored exactly as the official MAP@5.

Retrieval is a within-pair ranking problem, so absolute feature values matter
far less than a hand's standing inside its own pair. Variants tested:
  base   - current pointwise binary classifier
  rank   - listwise CatBoost ranker grouped by pair
  norm   - pointwise on within-pair z-scored features
  family - one ranker per disclosed behaviour family, routed by true family
           (an upper bound; inference must route by predicted family)
"""
import sys,json,argparse,time
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier,CatBoostRanker,Pool
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
ID={'pair_id','player_1','player_2','pool','fold','label','label_status','behavior_family','hand_id','hid','evidence_rank','ev'}
FAM=['directed_transfer','soft_play','coordinated_isolation']

def map5(df,score_col):
 tot=[]
 for (p,),g in df.group_by(['pair_id']):
  g=g.sort(score_col,descending=True)
  rel=g['ev'].to_numpy();k=int(rel.sum())
  if k==0:continue
  hits=0;s=0.
  for i,r in enumerate(rel[:5],1):
   if r:hits+=1;s+=hits/i
  tot.append(s/min(k,5))
 return float(np.mean(tot)),len(tot)

def znorm(x,groups):
 out=np.empty_like(x)
 for idx in groups.values():
  b=x[idx];m=b.mean(0);s=b.std(0);s[s<1e-6]=1
  out[idx]=(b-m)/s
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variants',default='base,rank,norm,family');args=ap.parse_args()
 h=pl.read_parquet('artifacts/dev_hands_policy.parquet').filter(pl.col('label')==1).sort('pair_id')
 fc=[c for c,t in h.schema.items() if c not in ID and t.is_numeric()]
 X=np.nan_to_num(h.select(fc).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 y=h['ev'].to_numpy();fold=h['fold'].to_numpy();pair=h['pair_id'].to_numpy();fam=h['behavior_family'].to_numpy()
 groups={}
 for i,p in enumerate(pair):groups.setdefault(p,[]).append(i)
 groups={k:np.array(v) for k,v in groups.items()}
 Z=znorm(X,groups)
 gid=np.searchsorted(np.unique(pair),pair)
 res={}
 for variant in args.variants.split(','):
  t=time.time();scores=np.zeros(len(y))
  for f in [1,2,3,4]:
   tr=(fold!=f)&(fold!=0);te=fold==f
   if variant=='base':
    m=CatBoostClassifier(iterations=500,depth=5,learning_rate=.055,l2_leaf_reg=8,thread_count=6,verbose=False,random_seed=2026,allow_writing_files=False)
    m.fit(X[tr],y[tr]);scores[te]=m.predict_proba(X[te])[:,1]
   elif variant=='norm':
    m=CatBoostClassifier(iterations=500,depth=5,learning_rate=.055,l2_leaf_reg=8,thread_count=6,verbose=False,random_seed=2026,allow_writing_files=False)
    m.fit(np.hstack([X,Z])[tr],y[tr]);scores[te]=m.predict_proba(np.hstack([X,Z])[te])[:,1]
   elif variant=='rank':
    XX=np.hstack([X,Z])
    o=np.argsort(gid[tr],kind='stable');itr=np.flatnonzero(tr)[o]
    m=CatBoostRanker(loss_function='YetiRank',iterations=600,depth=5,learning_rate=.06,thread_count=6,verbose=False,random_seed=2026,allow_writing_files=False)
    m.fit(Pool(XX[itr],y[itr],group_id=gid[itr]));scores[te]=m.predict(XX[te])
   elif variant=='family':
    XX=np.hstack([X,Z])
    for fm in FAM:
     trf=tr&(fam==fm);tef=te&(fam==fm)
     if tef.sum()==0:continue
     o=np.argsort(gid[trf],kind='stable');itr=np.flatnonzero(trf)[o]
     m=CatBoostRanker(loss_function='YetiRank',iterations=600,depth=5,learning_rate=.06,thread_count=6,verbose=False,random_seed=2026,allow_writing_files=False)
     m.fit(Pool(XX[itr],y[itr],group_id=gid[itr]));scores[tef]=m.predict(XX[tef])
  d=h.with_columns(pl.Series('s',scores))
  m5,n=map5(d.filter(pl.col('fold').is_in([1,2,3,4])),'s')
  per={fm:map5(d.filter(pl.col('fold').is_in([1,2,3,4])&(pl.col('behavior_family')==fm)),'s')[0] for fm in FAM}
  res[variant]={'map5':m5,'pairs':n,'per_family':per,'seconds':round(time.time()-t)}
  print(variant,json.dumps(res[variant]),flush=True)
 Path('reports/evidence_experiments.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
