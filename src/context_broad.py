"""Context features aggregated to pair level, for the RISK model.

Astra's action-local context features (immediate yielding to a partner, folding
a private pair or a draw, consistent giver/receiver roles) were built for
evidence retrieval and used only there. The pair risk model - 70% of the score -
has never seen them. They encode collusion mechanics directly, so aggregating
them per pair is the largest untried signal in the project.

Development needs the full candidate population, not just labelled pairs, so it
is extracted here. Evaluation already has per-hand context on disk from the v4
work and is only aggregated.
"""
from pathlib import Path
import argparse,time
import polars as pl
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from context_features import extract
from features import broad_pairs

def aggregate(df):
 fs=[c for c in df.columns if c.startswith('cx_')]
 ex=[pl.len().alias('cx_n')]
 for f in fs:
  ex.extend([pl.col(f).mean().alias(f+'__mean'),pl.col(f).max().alias(f+'__max'),
             pl.col(f).top_k(5).mean().alias(f+'__top5')])
 return df.group_by('pair_id').agg(ex)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',default='development')
 ap.add_argument('--start',type=int,default=0);ap.add_argument('--limit',type=int,default=400)
 args=ap.parse_args()
 out=Path(f'artifacts/context_agg_{args.phase}');out.mkdir(exist_ok=True,parents=True)
 t=time.time()
 if args.phase=='evaluation':
  # Per-hand context already exists for every evaluation pair; just aggregate.
  for i,pool in enumerate(range(400)):
   if i<args.start or i>=args.limit:continue
   p=out/f'{pool}.parquet'
   if p.exists():continue
   src=Path(f'artifacts/context_evaluation/{pool}.parquet')
   if not src.exists():continue
   aggregate(pl.read_parquet(src)).write_parquet(p)
   if i%50==0:print('eval agg',i,round(time.time()-t),flush=True)
 else:
  h=pl.read_parquet('artifacts/hands.parquet').filter(pl.col('phase')=='development')
  pairs=broad_pairs('development')
  for i,pool in enumerate(sorted(pairs['pool'].unique())):
   if i<args.start or i>=args.limit:continue
   p=out/f'{pool}.parquet'
   if p.exists():continue
   df=extract(pool,pairs.filter(pl.col('pool')==pool),h.filter(pl.col('pool')==pool))
   aggregate(df).write_parquet(p)
   if i%25==0:print('dev',i,'rows',df.height,round(time.time()-t),flush=True)
 print('DONE',round(time.time()-t),flush=True)
if __name__=='__main__':main()
