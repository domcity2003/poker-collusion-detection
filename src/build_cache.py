"""Materialise the two pair tables once, projected to the selected features.

pair_table() globs and joins ~1,200 parquet files; on this machine that is about
ten minutes and over a gigabyte of transient frames, paid again on every script
invocation. Everything downstream only ever needs the identifier columns plus a
few hundred features, so build it once and cache.
"""
import sys,json,time
from pathlib import Path
import polars as pl
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import pair_table,slim

def main():
 sel=json.loads(Path('artifacts/selected_features.json').read_text())
 cols=sel['top400']
 for phase in ['development','evaluation']:
  out=Path(f'artifacts/cache_{phase}.parquet')
  if out.exists():print('exists',out);continue
  t=time.time();df=slim(pair_table(phase),cols)
  df.write_parquet(out)
  print(phase,df.shape,round(time.time()-t),'s',flush=True)
if __name__=='__main__':main()
