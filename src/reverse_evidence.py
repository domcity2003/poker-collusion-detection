"""Search for the organisers' evidence-selection RULE rather than modelling it.

340 of 372 positive pairs list exactly five evidence hands, but 21 list four and
11 list three. A top-5-by-severity ranking would always yield five, so selection
must be a predicate with a threshold: a hand is evidence if it satisfies some
condition, capped at five. If the predicate can be recovered, retrieval stops
being a learning problem.

For each candidate predicate (feature > threshold) we measure how exactly the
qualifying set reproduces the listed evidence set, per behaviour family.
"""
import sys,json
from pathlib import Path
import numpy as np, polars as pl

def search(h,fam,cols,out):
 d=h.filter(pl.col('behavior_family')==fam)
 pairs=d.partition_by('pair_id')
 best=[]
 for c in cols:
  x=np.concatenate([p[c].to_numpy().astype(float) for p in pairs])
  if np.nanstd(x)==0:continue
  cand=[0.0]+[float(np.nanquantile(x,q)) for q in (.9,.95,.975,.99,.995)]
  for t in sorted(set(cand)):
   jac=[];exact=0;sizes=[];cov=[]
   for p in pairs:
    v=p[c].to_numpy().astype(float);e=p['ev'].to_numpy()==1
    s=v>t
    if s.sum()==0:continue
    inter=(s&e).sum();union=(s|e).sum()
    jac.append(inter/union);sizes.append(int(s.sum()));cov.append(inter/max(1,e.sum()))
    exact+=int(np.array_equal(s,e))
   if len(jac)<20:continue
   best.append({'feature':c,'threshold':t,'jaccard':float(np.mean(jac)),
                'recall':float(np.mean(cov)),'exact_sets':exact,'n_pairs':len(jac),
                'median_qualifying':float(np.median(sizes))})
 best.sort(key=lambda r:-r['jaccard'])
 print(f'\n=== {fam} (pairs={len(pairs)})')
 for r in best[:6]:
  print(f"  jac={r['jaccard']:.3f} recall={r['recall']:.3f} exact={r['exact_sets']}/{r['n_pairs']}"
        f" med|S|={r['median_qualifying']:.0f}  {r['feature']} > {r['threshold']:.4g}")
 out[fam]=best[:20]
 return best

def main():
 h=pl.read_parquet('artifacts/dev_hands_hindsight.parquet')
 skip={'pair_id','hand_id','hid','ev','label','fold','behavior_family','evidence_rank'}
 cols=[c for c,t in h.schema.items() if t.is_numeric() and c not in skip]
 print('candidate features:',len(cols))
 # Pairs listing fewer than five are the informative ones: the predicate must
 # produce exactly that many qualifying hands for them.
 n=h.group_by('pair_id').agg(pl.col('ev').sum().alias('k'))
 short=n.filter(pl.col('k')<5)
 print('pairs listing <5 evidence hands:',short.height)
 out={}
 for fam in ['directed_transfer','soft_play','coordinated_isolation']:
  search(h,fam,cols,out)
 Path('reports/reverse_evidence.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
