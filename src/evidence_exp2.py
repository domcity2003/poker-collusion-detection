"""Honest version of family-routed evidence retrieval.

evidence_exp.py's `family` variant routed by the TRUE family, so 0.494 is an
upper bound. At inference the family is predicted, and evaluation contains an
`other_coordination` family with no development examples at all. So instead of
hard routing, score every hand with all three specialists, rank-normalise each
specialist within the pair (YetiRank outputs are not comparable across models),
and blend by the pair's predicted family probability. A pair the classifier is
unsure about degrades towards the average of the three, not to a wrong one.
"""
import sys,json
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier,CatBoostRanker,Pool
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evidence_exp import map5,znorm,ID,FAM

def main():
 h=pl.read_parquet('artifacts/dev_hands_policy.parquet').filter(pl.col('label')==1).sort('pair_id')
 fc=[c for c,t in h.schema.items() if c not in ID and t.is_numeric()]
 X=np.nan_to_num(h.select(fc).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 y=h['ev'].to_numpy();fold=h['fold'].to_numpy();pair=h['pair_id'].to_numpy();fam=h['behavior_family'].to_numpy()
 groups={}
 for i,p in enumerate(pair):groups.setdefault(p,[]).append(i)
 groups={k:np.array(v) for k,v in groups.items()}
 XX=np.hstack([X,znorm(X,groups)])
 gid=np.searchsorted(np.unique(pair),pair)
 # Pair-level behaviour probabilities, cross-fitted so a pair never sees its own label.
 pt=pl.read_parquet('artifacts/development_broad/*.agg.parquet')
 pt=pt.with_columns(pl.col('pair_id').str.split('|').list.get(0).alias('player_1'),
                    pl.col('pair_id').str.split('|').list.get(1).alias('player_2')).drop('pair_id')
 lab=pl.read_parquet('artifacts/labels.parquet')
 pt=lab.join(pt,on=['player_1','player_2']).filter(pl.col('label')==1)
 pc=[c for c,t in pt.schema.items() if c not in ID and t.is_numeric()]
 PX=np.nan_to_num(pt.select(pc).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 py=np.array([FAM.index(x) for x in pt['behavior_family']]);pf=pt['fold'].to_numpy()
 prob={}
 for f in [1,2,3,4]:
  m=CatBoostClassifier(iterations=400,depth=4,learning_rate=.06,thread_count=6,verbose=False,random_seed=2026,allow_writing_files=False)
  m.fit(PX[(pf!=f)&(pf!=0)],py[(pf!=f)&(pf!=0)])
  for pid,p in zip(pt.filter(pl.col('fold')==f)['pair_id'],m.predict_proba(PX[pf==f])):prob[pid]=p
 acc=np.mean([FAM[int(np.argmax(prob[p]))]==f for p,f in zip(pt['pair_id'],pt['behavior_family']) if p in prob])
 print('family accuracy (cross-fitted):',round(float(acc),4),flush=True)
 scores=np.zeros(len(y))
 for f in [1,2,3,4]:
  tr=(fold!=f)&(fold!=0);te=np.flatnonzero(fold==f)
  spec=[]
  for fm in FAM:
   trf=tr&(fam==fm);o=np.argsort(gid[trf],kind='stable');itr=np.flatnonzero(trf)[o]
   m=CatBoostRanker(loss_function='YetiRank',iterations=600,depth=5,learning_rate=.06,thread_count=6,verbose=False,random_seed=2026,allow_writing_files=False)
   m.fit(Pool(XX[itr],y[itr],group_id=gid[itr]));spec.append(m.predict(XX[te]))
  spec=np.vstack(spec)
  tep=pair[te]
  for p,idx in ((p,np.flatnonzero(tep==p)) for p in np.unique(tep)):
   w=prob.get(p,np.ones(3)/3)
   block=spec[:,idx]
   r=np.vstack([(np.argsort(np.argsort(b))+1)/len(b) for b in block])
   scores[te[idx]]=(w[:,None]*r).sum(0)
 d=h.with_columns(pl.Series('s',scores)).filter(pl.col('fold').is_in([1,2,3,4]))
 m5,n=map5(d,'s')
 per={fm:map5(d.filter(pl.col('behavior_family')==fm),'s')[0] for fm in FAM}
 out={'variant':'family_blend_predicted','map5':m5,'pairs':n,'per_family':per,'family_accuracy':float(acc)}
 print(json.dumps(out),flush=True)
 Path('reports/evidence_family_blend.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
