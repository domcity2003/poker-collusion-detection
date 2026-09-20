"""Is the coordinated_isolation ranker starved of data?

Each evidence specialist trains only on its own family's pairs. For
directed_transfer that is 118 pairs and for soft_play 107, but isolation has
just 79 - and isolation is the family stuck at MAP@5 .27 while the others reach
.56 and .63. A specialist that thin may simply be over-fitting, in which case a
ranker trained on every positive pair (more data, less family-specific) can beat
it on isolation even though it is less targeted.

Tested with the same fold discipline as everything else: train on folds != test,
fold 0 never touched.
"""
import sys,json
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostRanker,Pool
sys.path.insert(0,str(Path(__file__).resolve().parent))
from final import mat,znorm,grouping
from pipeline import ID,FAM

def ap5(rel):
 k=int(rel.sum())
 if k==0:return None
 hits=0;s=0.
 for i,r in enumerate(rel[:5],1):
  if r:hits+=1;s+=hits/i
 return s/min(k,5)

def score_family(h,scores,family,pairs_idx):
 rank=lambda x:(np.argsort(np.argsort(x))+1)/len(x)
 ev=h['ev'].to_numpy();vals=[]
 for idx in pairs_idx:
  a=ap5(ev[idx][np.argsort(-scores[idx])])
  if a is not None:vals.append(a)
 return float(np.mean(vals)),len(vals)

def main():
 h=pl.read_parquet('artifacts/dev_hands_context.parquet').sort('pair_id')
 fc=[c for c,t in h.schema.items() if c not in ID and t.is_numeric()]
 X=mat(h,fc);pair=h['pair_id'].to_numpy()
 XX=np.hstack([X,znorm(X,grouping(pair))]);del X
 gid=np.searchsorted(np.unique(pair),pair)
 y=h['ev'].to_numpy();fam=h['behavior_family'].to_numpy();fold=h['fold'].to_numpy()
 lab=h['label'].to_numpy()
 sel=np.isin(fold,[1,2,3,4])&(lab==1)
 def rk(name,restrict):
  s=np.zeros(len(y))
  for f in [1,2,3,4]:
   tr=(fold!=f)&(fold!=0)&(lab==1)&restrict;te=np.flatnonzero(fold==f)
   i=np.flatnonzero(tr);i=i[np.argsort(gid[i],kind='stable')]
   m=CatBoostRanker(loss_function='YetiRank',iterations=600,depth=5,learning_rate=.06,
     thread_count=6,verbose=False,random_seed=2026,allow_writing_files=False)
   m.fit(Pool(XX[i],y[i],group_id=gid[i]));s[te]=m.predict(XX[te])
  np.save(f'artifacts/context_cv/pooled_{name}.npy',s)
  print('fitted',name,flush=True);return s
 iso=rk('iso_specialist',fam=='coordinated_isolation')
 allf=rk('all_families',np.ones(len(y),bool))
 g={}
 for i in np.flatnonzero(sel&(fam=='coordinated_isolation')):g.setdefault(pair[i],[]).append(i)
 idxs=[np.array(v) for v in g.values()]
 base=np.load('artifacts/context_cv/baseline.npy')[2]
 ctx=np.load('artifacts/context_cv/context.npy')[2]
 rank=lambda x:(np.argsort(np.argsort(x))+1)/len(x)
 def blend(*parts):
  out=np.zeros(len(y))
  for idx in idxs:
   out[idx]=sum(w*rank(p[idx]) for w,p in parts)
  return out
 res={}
 for name,s in [('v4 blend (base+context)',blend((.5,base),(.5,ctx))),
                ('isolation specialist refit',blend((1.,iso))),
                ('all-family ranker',blend((1.,allf))),
                ('base+allfamily',blend((.5,base),(.5,allf))),
                ('base+context+allfamily',blend((1/3,base),(1/3,ctx),(1/3,allf)))]:
  m,n=score_family(h,s,'coordinated_isolation',idxs);res[name]=m
  print(f'  {name:32} isolation MAP@5 {m:.4f}  (n={n})',flush=True)
 Path('reports/isolation_pool.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
