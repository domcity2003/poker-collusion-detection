"""Post-selection stress check on previously-used fold 0; not a fresh holdout."""
import json
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier,CatBoostRanker,Pool
from pipeline import ID,FAM,feature_columns
from final import mat,znorm,grouping
from context_experiment import metrics
h=pl.read_parquet('artifacts/dev_hands_context.parquet');h=h.sort('pair_id')
fc=[c for c,t in h.schema.items() if c not in ID and t.is_numeric()]
y=h['ev'].to_numpy();pair=h['pair_id'].to_numpy();fold=h['fold'].to_numpy();fam=h['behavior_family'].to_numpy();gid=np.searchsorted(np.unique(pair),pair);g=grouping(pair);te=np.flatnonzero(fold==0)
lab=pl.read_parquet('artifacts/labels.parquet');pt=pl.scan_parquet('artifacts/cache_development.parquet').filter(pl.col('label')==1).collect().drop('pair_id')
pt=pt.join(lab.select('player_1','player_2','pair_id'),on=['player_1','player_2']);pc=feature_columns(pt)
px=mat(pt,pc);pf=pt['fold'].to_numpy();py=np.array([FAM.index(x) for x in pt['behavior_family']])
m=CatBoostClassifier(iterations=400,depth=4,learning_rate=.06,thread_count=5,verbose=False,random_seed=2026,allow_writing_files=False).fit(px[pf!=0],py[pf!=0]);probs=dict(zip(pt.filter(pl.col('fold')==0)['pair_id'],m.predict_proba(px[pf==0])))
preds={}
for tag in ['baseline','context']:
 cols=[c for c in fc if tag!='baseline' or not c.startswith('cx_')];X=mat(h,cols);XX=np.hstack([X,znorm(X,g)]);s=np.zeros((3,len(y)))
 for k,fm in enumerate(FAM):
  tr=np.flatnonzero((fold!=0)&(fam==fm));m=CatBoostRanker(loss_function='YetiRank',iterations=600,depth=5,learning_rate=.06,thread_count=5,verbose=False,random_seed=2026,allow_writing_files=False)
  m.fit(Pool(XX[tr],y[tr],group_id=gid[tr]));s[k,te]=m.predict(XX[te])
 for p,idx in g.items():s[:,idx]=np.vstack([(np.argsort(np.argsort(x))+1)/len(x) for x in s[:,idx]])
 preds[tag]=s
res={}
for tag,spec in [('baseline',preds['baseline']),('selected',preds['baseline']*np.array([.5,1,.5])[:,None]+preds['context']*np.array([.5,0,.5])[:,None])]:
 scores=np.zeros(len(y))
 for p,idx in g.items():scores[idx]=(np.array(probs.get(p,np.ones(3)/3))[:,None]*spec[:,idx]).sum(0)
 # metrics excludes fold0, so relabel only for this reporting function.
 test=h.filter(pl.col('fold')==0).with_columns(pl.lit(9).alias('fold'));d=metrics(test,scores[te]);res[tag]={'map5':d['ap'].mean(),'family':d.group_by('family').agg(pl.col('ap').mean()).to_dicts()}
print(json.dumps(res,indent=2),flush=True);Path('reports/context_holdout.json').write_text(json.dumps(res,indent=2))
