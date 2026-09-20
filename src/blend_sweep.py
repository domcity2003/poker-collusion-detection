"""Per-family evidence blend weight, swept on saved out-of-fold scores.

v4 froze a single 0.5/0.5 rank blend for directed_transfer and
coordinated_isolation and left soft_play on the baseline ranker alone, because
the blend hurt soft_play at 0.5. That is three families sharing one weight
chosen from one comparison. The out-of-fold ranker scores are already on disk,
so the whole weight curve costs no model fitting at all.

Blending is done in within-pair rank space, exactly as the submission builder
does it, so a weight chosen here transfers unchanged.
"""
import sys,json
from pathlib import Path
import numpy as np, polars as pl
FAM=['directed_transfer','soft_play','coordinated_isolation']

def ap5(rel):
 k=int(rel.sum())
 if k==0:return None
 hits=0;s=0.
 for i,r in enumerate(rel[:5],1):
  if r:hits+=1;s+=hits/i
 return s/min(k,5)

def main():
 h=pl.read_parquet('artifacts/dev_hands_context.parquet').sort('pair_id')
 b=np.load('artifacts/context_cv/baseline.npy');c=np.load('artifacts/context_cv/context.npy')
 fold=h['fold'].to_numpy();fam=h['behavior_family'].to_numpy()
 pair=h['pair_id'].to_numpy();ev=h['ev'].to_numpy()
 sel=np.isin(fold,[1,2,3,4])&(h['label'].to_numpy()==1)
 groups={}
 for i in np.flatnonzero(sel):groups.setdefault(pair[i],[]).append(i)
 groups={k:np.array(v) for k,v in groups.items()}
 rank=lambda x:(np.argsort(np.argsort(x))+1)/len(x)
 W=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
 out={};print(f"{'family':24}"+"".join(f'{w:>8}' for w in W))
 for fi,f in enumerate(FAM):
  ids=[p for p,idx in groups.items() if fam[idx[0]]==f]
  row=[]
  for w in W:
   vals=[]
   for p in ids:
    idx=groups[p]
    s=(1-w)*rank(b[fi][idx])+w*rank(c[fi][idx])
    a=ap5(ev[idx][np.argsort(-s)])
    if a is not None:vals.append(a)
   row.append(float(np.mean(vals)))
  out[f]={'n_pairs':len(ids),'weights':W,'map5':row,
          'best_w':W[int(np.argmax(row))],'best':max(row),'w0':row[0],'w50':row[5]}
  print(f'{f:24}'+"".join(f'{v:8.4f}' for v in row))
 # Overall MAP@5 under a chosen per-family weight vector, all families pooled.
 def overall(ws):
  vals=[]
  for p,idx in groups.items():
   fi=FAM.index(fam[idx[0]]);w=ws[fi]
   s=(1-w)*rank(b[fi][idx])+w*rank(c[fi][idx])
   a=ap5(ev[idx][np.argsort(-s)])
   if a is not None:vals.append(a)
  return float(np.mean(vals))
 v4=[.5,0,.5];best=[out[f]['best_w'] for f in FAM]
 res={'per_family':out,'overall_v4_weights':overall(v4),'v4_weights':v4,
      'overall_best_weights':overall(best),'best_weights':best,
      'overall_all_zero':overall([0,0,0])}
 print(f"\noverall MAP@5  baseline-only {res['overall_all_zero']:.4f}"
       f"  |  v4 weights {v4} -> {res['overall_v4_weights']:.4f}"
       f"  |  tuned {best} -> {res['overall_best_weights']:.4f}")
 Path('reports/blend_sweep.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
