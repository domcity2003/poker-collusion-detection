"""Seed-average the evidence rankers.

Every evidence ranker so far is a single seed (2026). YetiRank on 79-118 pairs
per family is a high-variance fit, and averaging seeds is the one improvement
that needs no new features, no new data and cannot overfit the validation set -
it only removes noise. Measured per family so it can be combined with the
per-family blend weights from blend_sweep.py.
"""
import sys,json,time
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostRanker,Pool
sys.path.insert(0,str(Path(__file__).resolve().parent))
from final import mat,znorm,grouping
from pipeline import ID,FAM
SEEDS=(2026,7,991)

def ap5(rel):
 k=int(rel.sum())
 if k==0:return None
 hits=0;s=0.
 for i,r in enumerate(rel[:5],1):
  if r:hits+=1;s+=hits/i
 return s/min(k,5)

def main():
 h=pl.read_parquet('artifacts/dev_hands_context.parquet').sort('pair_id')
 fc=[c for c,t in h.schema.items() if c not in ID and t.is_numeric()]
 bfc=json.loads(Path('artifacts/recovered_v2/columns.json').read_text())['hand']
 pair=h['pair_id'].to_numpy();gid=np.searchsorted(np.unique(pair),pair)
 g=grouping(pair)
 y=h['ev'].to_numpy();fam=h['behavior_family'].to_numpy();fold=h['fold'].to_numpy();lab=h['label'].to_numpy()
 t=time.time()
 out={}
 for tag,cols in [('base',bfc),('ctx',fc)]:
  X=mat(h,cols);XX=np.hstack([X,znorm(X,g)]);del X
  for fi,f in enumerate(FAM):
   if tag=='ctx' and f=='soft_play':continue   # v4 never uses context for soft_play
   s=np.zeros((len(SEEDS),len(y)))
   for si,sd in enumerate(SEEDS):
    for fo in [1,2,3,4]:
     tr=(fold!=fo)&(fold!=0)&(lab==1)&(fam==f);te=np.flatnonzero(fold==fo)
     i=np.flatnonzero(tr);i=i[np.argsort(gid[i],kind='stable')]
     m=CatBoostRanker(loss_function='YetiRank',iterations=600,depth=5,learning_rate=.06,
       thread_count=6,verbose=False,random_seed=sd,allow_writing_files=False)
     m.fit(Pool(XX[i],y[i],group_id=gid[i]));s[si,te]=m.predict(XX[te])
    print(f'  {tag} {f} seed {sd} done {round(time.time()-t)}s',flush=True)
   np.save(f'artifacts/context_cv/seeds_{tag}_{fi}.npy',s)
   out[f'{tag}_{f}']='saved'
  del XX
 print('ALL FITTED',round(time.time()-t),flush=True)
if __name__=='__main__':main()
