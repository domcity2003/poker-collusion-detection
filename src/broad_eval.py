"""Realistic pair ranking check: positives vs the whole in-fold candidate population.

Development CV over labeled pairs only is saturated (AP ~0.99) because confirmed
negatives are easy. The real evaluation ranks 112,540 pairs of which the vast
majority are undisclosed. Treating every unlabeled candidate as a negative is
pessimistic (some are truly positive) but it is the only lower bound we can
measure, and it moves when the risk model actually improves.
"""
import sys,json
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import _average_precision

def main():
 labels=pl.read_parquet('artifacts/labels.parquet')
 rows=[]
 for f in [1,2,3,4]:
  p=pl.read_parquet(f'artifacts/models/population_oof_{f}.parquet')
  y=(p['label'].to_numpy()==1).astype(int);s=p['risk'].to_numpy()
  known=p['label'].to_numpy()>=0
  rows.append({'fold':f,'n_candidates':p.height,'n_pos':int(y.sum()),
   'broad_ap':_average_precision(y,s),
   'labeled_only_ap':_average_precision(y[known],s[known]),
   'recall_at_1pct':float(y[np.argsort(-s)[:max(1,p.height//100)]].sum()/max(1,y.sum())),
   'unknown_above_median_pos':float((s[~known]>np.median(s[y==1])).mean())})
 allp=pl.concat([pl.read_parquet(f'artifacts/models/population_oof_{f}.parquet') for f in [1,2,3,4]])
 y=(allp['label'].to_numpy()==1).astype(int);s=allp['risk'].to_numpy()
 pooled={'n_candidates':allp.height,'n_pos':int(y.sum()),'broad_ap':_average_precision(y,s)}
 # How many confirmed positives are even present in the candidate population?
 pos=labels.filter(pl.col('label')==1)
 cov=allp.filter(pl.col('label')==1).height/pos.filter(pl.col('fold')!=0).height
 pooled['positive_coverage_folds1_4']=cov
 out={'folds':rows,'pooled':pooled}
 Path('reports/broad_validation.json').write_text(json.dumps(out,indent=2,default=float))
 print(json.dumps(out,indent=2,default=float))
if __name__=='__main__':main()
