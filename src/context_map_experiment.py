"""Paired pool-held-out evidence test: frozen baseline versus new action context."""
import json,pickle,time
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostRanker,Pool
from final import mat,znorm,grouping
from pipeline import ID,FAM

def metrics(h,s):
 out=[]
 for (p,),g in h.with_columns(pl.Series('score',s)).group_by('pair_id'):
  if g['fold'][0]==0:continue
  y=g.sort('score',descending=True)['ev'].to_numpy();hit=np.cumsum(y[:5]);ap=float((hit/np.arange(1,min(5,len(y))+1)*y[:5]).sum()/min(y.sum(),5))
  out.append({'pair_id':p,'pool':int(g['pool'][0]),'fold':int(g['fold'][0]),'family':g['behavior_family'][0],'ap':ap})
 return pl.DataFrame(out)

def main():
 h=pl.read_parquet('artifacts/dev_hands_hindsight.parquet').filter(pl.col('label')==1).sort('pair_id')
 cx=pl.read_parquet('artifacts/context_development/*.parquet')
 h=h.join(cx,on=['pair_id','hand_id'],how='left',maintain_order='left');assert h['cx_yield_immediate'].null_count()==0
 h.write_parquet('artifacts/dev_hands_context.parquet')
 fc=[c for c,t in h.schema.items() if c not in ID and t.is_numeric()]
 probs=pickle.load(open('artifacts/fam_probs.pkl','rb'))
 y=h['ev'].to_numpy();fold=h['fold'].to_numpy();pair=h['pair_id'].to_numpy();fam=h['behavior_family'].to_numpy();g=grouping(pair);gid=np.searchsorted(np.unique(pair),pair)
 out=Path('artifacts/context_cv');out.mkdir(exist_ok=True);res={};preds={}
 for tag in ['baseline','map5']:
  t=time.time();cols=[c for c in fc if not c.startswith('cx_')]
  path=out/f'{tag}.npy'
  if path.exists():scores=np.load(path)
  else:
   X=mat(h,cols);XX=np.hstack([X,znorm(X,g)]);scores=np.zeros((3,len(y)))
   for f in [1,2,3,4]:
    te=np.flatnonzero(fold==f);tr=(fold!=f)&(fold!=0)
    for k,fm in enumerate(FAM):
     idx=np.flatnonzero(tr&(fam==fm))
     m=CatBoostRanker(loss_function='YetiRank:mode=MAP;top=5' if tag=='map5' else 'YetiRank',iterations=600,depth=5,learning_rate=.06,thread_count=5,verbose=False,random_seed=2026,allow_writing_files=False)
     m.fit(Pool(XX[idx],y[idx],group_id=gid[idx]));scores[k,te]=m.predict(XX[te])
    print(tag,'fold',f,round(time.time()-t),flush=True)
   np.save(path,scores)
  # Percentile specialist scores exactly match production rank mixing.
  rank=np.zeros_like(scores)
  for p,idx in g.items():rank[:,idx]=np.vstack([(np.argsort(np.argsort(x))+1)/len(x) for x in scores[:,idx]])
  preds[tag]=rank
 for tag,rank in [('baseline',preds['baseline']),('map5',preds['map5']),('blend50',.5*preds['map5']+.5*preds['baseline'])]:
  s=np.zeros(len(y))
  for p,idx in g.items():s[idx]=(np.asarray(probs.get(p,np.ones(3)/3))[:,None]*rank[:,idx]).sum(0)
  m=metrics(h,s);m.write_csv(out/f'{tag}_pair_metrics.csv')
  res[tag]={'map5':float(m['ap'].mean()),'folds':m.group_by('fold').agg(pl.col('ap').mean()).sort('fold').to_dicts(),'family':m.group_by('family').agg(pl.col('ap').mean()).to_dicts()}
  print(tag,json.dumps(res[tag]),flush=True)
 Path('reports/context_map_experiment.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
