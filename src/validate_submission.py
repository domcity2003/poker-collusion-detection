"""Hard checks before a submission slot is spent.

Beyond the scorer's own schema rules this verifies the thing the scorer cannot:
that every cited evidence hand is a hand the pair actually shared during the
evaluation phase. A hand id that is real but belongs to another pair or the
development phase would silently score zero.
"""
import sys,hashlib
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import score as official_score,ALLOWED_BEHAVIORS,EVIDENCE_COLUMNS

def main(path='artifacts/final/submission.csv'):
 sub=pd.read_csv(path);fail=[]
 pairs=pd.read_csv('data/evaluation_pairs.csv');sample=pd.read_csv('data/sample_submission.csv')
 def chk(c,msg):
  print(('PASS ' if c else 'FAIL ')+msg)
  if not c:fail.append(msg)
 chk(len(sub)==112540,f'row count 112540 (got {len(sub)})')
 chk(set(sub.pair_id)==set(sample.pair_id),'pair_id set matches sample_submission')
 chk(not sub.pair_id.duplicated().any(),'pair_id unique')
 chk(list(sub.columns)==list(sample.columns),f'column order matches sample {list(sample.columns)} vs {list(sub.columns)}')
 r=pd.to_numeric(sub.risk_score,errors='coerce')
 chk(r.notna().all() and r.between(0,1).all(),'risk_score numeric in [0,1]')
 chk(r.nunique()>1000,f'risk_score not degenerate ({r.nunique()} distinct)')
 chk(set(sub.predicted_behavior)<=ALLOWED_BEHAVIORS,f'behaviors allowed {set(sub.predicted_behavior)}')
 ev=sub[list(EVIDENCE_COLUMNS)]
 chk(ev.notna().all().all(),'no null evidence cells')
 dup=(ev.apply(lambda row:len(set(row))!=len(row),axis=1)).sum()
 chk(dup==0,f'no repeated evidence within a pair ({dup} offenders)')
 chk((ev=='NO_EVIDENCE').sum().sum()==0,'every pair cites five hands')
 # Evidence must be a hand this pair truly shared in the evaluation phase.
 h=pl.read_parquet('artifacts/hands.parquet').filter(pl.col('phase')=='evaluation')
 valid=set(h['hand_id'].to_list())
 cited=set(pd.unique(ev.values.ravel()))
 chk(cited<=valid,f'all cited hands are evaluation hands ({len(cited-valid)} unknown)')
 sm=sub.melt('pair_id',list(EVIDENCE_COLUMNS),value_name='hand_id')[['pair_id','hand_id']]
 truth=[]
 for pool in range(400):
  p=Path(f'artifacts/evaluation/{pool}.parquet')
  if p.exists():truth.append(pl.read_parquet(p).select('pair_id','hand_id').unique())
 truth=pl.concat(truth)
 j=pl.from_pandas(sm).join(truth.with_columns(pl.lit(1).alias('ok')),on=['pair_id','hand_id'],how='left')
 bad=int(j['ok'].is_null().sum())
 chk(bad==0,f'every cited hand was shared by that pair ({bad} bad citations)')
 # Scorer must accept the file against a schema-valid dummy solution.
 sol=pd.DataFrame({'pair_id':sub.pair_id,'risk_score':np.zeros(len(sub),dtype=int),
                   'predicted_behavior':'none'})
 for c in EVIDENCE_COLUMNS:sol[c]='NO_EVIDENCE'
 try:
  official_score(sol,sub.copy(),'pair_id');chk(True,'official scorer accepts the file')
 except Exception as e:chk(False,f'official scorer rejected: {e}')
 print('\nsha256',hashlib.sha256(Path(path).read_bytes()).hexdigest())
 print('behavior mix\n',sub.predicted_behavior.value_counts().to_string())
 print('risk quantiles',[round(float(r.quantile(q)),4) for q in (.5,.9,.99,.999,1)])
 print('\nRESULT:','READY' if not fail else f'{len(fail)} FAILURES')
 return 1 if fail else 0
if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
