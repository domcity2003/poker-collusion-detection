"""Build v4: verified-v2 risk and behavior; selective evidence ensemble only.
Run only after context CV is accepted and evaluation context extraction complete.
"""
from pathlib import Path
import json,hashlib,gc
import numpy as np,pandas as pd,polars as pl
from catboost import CatBoostRanker,Pool
from final import mat,znorm,grouping
from pipeline import ID,FAM,hand_table
OUT=Path('artifacts/v4');OUT.mkdir(exist_ok=True)

def fit():
 h=pl.read_parquet('artifacts/dev_hands_context.parquet').sort('pair_id')
 fc=[c for c,t in h.schema.items() if c not in ID and t.is_numeric()]
 X=mat(h,fc);pair=h['pair_id'].to_numpy();XX=np.hstack([X,znorm(X,grouping(pair))]);gid=np.searchsorted(np.unique(pair),pair);fam=h['behavior_family'].to_numpy();y=h['ev'].to_numpy()
 models={}
 for k in [0,2]:
  path=OUT/f'context_{k}.cbm';m=CatBoostRanker(loss_function='YetiRank',iterations=600,depth=5,learning_rate=.06,thread_count=5,verbose=False,random_seed=2026,allow_writing_files=False)
  if path.exists():m.load_model(str(path))
  else:
   ix=np.flatnonzero(fam==FAM[k]);m.fit(Pool(XX[ix],y[ix],group_id=gid[ix]));m.save_model(str(path))
  models[k]=m
 (OUT/'hand_columns.json').write_text(json.dumps(fc))
 return models,fc

def main():
 models,fc=fit();gc.collect()
 bfc=json.loads(Path('artifacts/recovered_v2/columns.json').read_text())['hand'];base=[]
 for i in range(3):m=CatBoostRanker();m.load_model(f'artifacts/recovered_v2/evidence_{i}.cbm');base.append(m)
 # Preserve original strings, including exact serialized risk scores.
 original=pd.read_csv('artifacts/final/submission_v2_verified.csv',dtype=str);sub=original.copy().set_index('pair_id')
 pr=pl.read_parquet('artifacts/recovered_v2/predictions.parquet');prob={r['pair_id']:np.array([r[f'family_{i}'] for i in range(3)]) for r in pr.iter_rows(named=True)}
 pairs=pl.read_csv('data/evaluation_pairs.csv').join(pl.read_parquet('artifacts/player_pools.parquet'),left_on='player_1',right_on='player_id')
 changed=0
 for pool in range(400):
  path=OUT/f'top_{pool}.parquet'
  if path.exists():top=pl.read_parquet(path)
  else:
   pp=pairs.filter(pl.col('pool')==pool);ids=[p for p in pp['pair_id'] if sub.loc[p,'predicted_behavior']!='soft_play'];pp=pp.filter(pl.col('pair_id').is_in(ids))
   if pp.height==0:continue
   h=hand_table('evaluation',pool,pp).join(pl.read_parquet(f'artifacts/context_evaluation/{pool}.parquet'),on=['pair_id','hand_id'],how='left',maintain_order='left')
   assert h['cx_yield_immediate'].null_count()==0
   p=h['pair_id'].to_numpy();hid=h['hand_id'].to_numpy();g=grouping(p);X=mat(h,bfc);XX=np.hstack([X,znorm(X,g)]);bs=np.vstack([m.predict(XX) for m in base]);del X,XX
   X=mat(h,fc);XX=np.hstack([X,znorm(X,g)]);cs={k:m.predict(XX) for k,m in models.items()};del X,XX
   rows=[]
   for pid,idx in g.items():
    r=np.vstack([(np.argsort(np.argsort(x))+1)/len(x) for x in bs[:,idx]])
    for k in [0,2]:
     x=cs[k][idx];r[k]=.5*r[k]+.5*(np.argsort(np.argsort(x))+1)/len(x)
    w=prob[pid];fm=FAM.index(sub.loc[pid,'predicted_behavior'])
    if w.argmax()!=fm:w=np.eye(3)[fm]
    rank=(w[:,None]*r).sum(0);chosen=hid[idx][np.argsort(-rank)[:5]]
    rows.append({'pair_id':pid,**{f'evidence_hand_{i+1}':v for i,v in enumerate(chosen)}})
   top=pl.DataFrame(rows);top.write_parquet(path)
  for r in top.iter_rows(named=True):
   pid=r.pop('pair_id')
   for k,v in r.items():sub.loc[pid,k]=v
  if pool%50==0:print('evidence pool',pool,flush=True)
 sub=sub.loc[original.pair_id].reset_index();assert sub[['pair_id','risk_score','predicted_behavior']].equals(original[['pair_id','risk_score','predicted_behavior']])
 ec=[f'evidence_hand_{i}' for i in range(1,6)];mask=original.predicted_behavior=='soft_play';assert sub.loc[mask,ec].equals(original.loc[mask,ec])
 dest=OUT/'submission_v4.csv';sub.to_csv(dest,index=False)
 info={'baseline_sha256':hashlib.sha256(Path('artifacts/final/submission_v2_verified.csv').read_bytes()).hexdigest(),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'rows':len(sub),'changed_evidence_rows':int((sub[ec]!=original[ec]).any(axis=1).sum()),'risk_and_behavior_exactly_preserved':True,'soft_play_rows_exactly_preserved':True,'context_weights':[.5,0,.5]}
 (OUT/'manifest.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
if __name__=='__main__':main()
