"""v6: evidence-only, per-family blend weights. Single variable against v5.

v4 used one blend weight (0.5) for directed_transfer and coordinated_isolation
and 0 for soft_play. Sweeping the weight on the saved out-of-fold scores (no
refitting - see blend_sweep.py) shows the three families want different weights:

  directed_transfer      .5 -> .7   MAP@5 .5697 -> .5771   smooth, real
  soft_play              0 stays    curve decreases monotonically from .6316
  coordinated_isolation  .5 stays   .270-.287 with no shape, i.e. noise on 79
                                    pairs - deliberately NOT tuned

Overall CV MAP@5 .5171 -> .5199. Small, but the only change this round that
survived scrutiny: seed-averaging the rankers measurably HURT (.5171 -> .5112,
soft_play .632 -> .617) and pooling families for isolation hurt more.

Everything except the evidence columns is copied byte-for-byte from v5.
"""
from pathlib import Path
import json,hashlib,gc
import numpy as np,pandas as pd,polars as pl
from catboost import CatBoostRanker
from final import mat,znorm,grouping
from pipeline import ID,FAM,hand_table
OUT=Path('artifacts/v6');OUT.mkdir(exist_ok=True)
W={0:.7,1:0.,2:.5}          # per-family weight on the context ranker

def main():
 fc=json.loads(Path('artifacts/v4/hand_columns.json').read_text())
 bfc=json.loads(Path('artifacts/recovered_v2/columns.json').read_text())['hand']
 base=[]
 for i in range(3):
  m=CatBoostRanker();m.load_model(f'artifacts/recovered_v2/evidence_{i}.cbm');base.append(m)
 ctx={}
 for k in (0,2):
  m=CatBoostRanker();m.load_model(f'artifacts/v4/context_{k}.cbm');ctx[k]=m
 original=pd.read_csv('artifacts/v5/submission_v5.csv',dtype=str)
 sub=original.copy().set_index('pair_id')
 pr=pl.read_parquet('artifacts/recovered_v2/predictions.parquet')
 prob={r['pair_id']:np.array([r[f'family_{i}'] for i in range(3)]) for r in pr.iter_rows(named=True)}
 pairs=pl.read_csv('data/evaluation_pairs.csv').join(
   pl.read_parquet('artifacts/player_pools.parquet'),left_on='player_1',right_on='player_id')
 for pool in range(400):
  path=OUT/f'top_{pool}.parquet'
  if path.exists():top=pl.read_parquet(path)
  else:
   pp=pairs.filter(pl.col('pool')==pool)
   if pp.height==0:continue
   h=hand_table('evaluation',pool,pp).join(
     pl.read_parquet(f'artifacts/context_evaluation/{pool}.parquet'),
     on=['pair_id','hand_id'],how='left',maintain_order='left')
   assert h['cx_yield_immediate'].null_count()==0
   p=h['pair_id'].to_numpy();hid=h['hand_id'].to_numpy();g=grouping(p)
   X=mat(h,bfc);XX=np.hstack([X,znorm(X,g)]);bs=np.vstack([m.predict(XX) for m in base]);del X,XX
   X=mat(h,fc);XX=np.hstack([X,znorm(X,g)]);cs={k:m.predict(XX) for k,m in ctx.items()};del X,XX
   gc.collect()
   rows=[]
   for pid,idx in g.items():
    r=np.vstack([(np.argsort(np.argsort(x))+1)/len(x) for x in bs[:,idx]])
    for k,w in W.items():
     if w>0:
      x=cs[k][idx];r[k]=(1-w)*r[k]+w*(np.argsort(np.argsort(x))+1)/len(x)
    w=prob[pid];fm=FAM.index(sub.loc[pid,'predicted_behavior'])
    if w.argmax()!=fm:w=np.eye(3)[fm]
    chosen=hid[idx][np.argsort(-(w[:,None]*r).sum(0))[:5]]
    rows.append({'pair_id':pid,**{f'evidence_hand_{i+1}':v for i,v in enumerate(chosen)}})
   top=pl.DataFrame(rows);top.write_parquet(path)
  for r in top.iter_rows(named=True):
   pid=r.pop('pair_id')
   for k,v in r.items():sub.loc[pid,k]=v
  if pool%100==0:print('evidence pool',pool,flush=True)
 sub=sub.loc[original.pair_id].reset_index()
 keep=['pair_id','risk_score','predicted_behavior']
 assert sub[keep].equals(original[keep]),'risk or behaviour drifted'
 out=OUT/'submission_v6.csv';sub.to_csv(out,index=False)
 ec=[f'evidence_hand_{i}' for i in range(1,6)]
 changed=int((sub[ec]!=original[ec]).any(axis=1).sum())
 man={'baseline':'artifacts/v5/submission_v5.csv',
      'baseline_sha256':hashlib.sha256(Path('artifacts/v5/submission_v5.csv').read_bytes()).hexdigest(),
      'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'weights':W,
      'only_evidence_changed':True,'changed_evidence_rows':changed,'rows':len(sub)}
 (OUT/'manifest.json').write_text(json.dumps(man,indent=2));print(json.dumps(man,indent=2),flush=True)
if __name__=='__main__':main()
