"""Does seed-averaging the evidence rankers help, and by how much per family?

Reads the per-seed out-of-fold scores written by evidence_seeds.py and compares
one seed against the average of three, at each family's blend weight. Averaging
is applied in within-pair rank space, the same space the submission blends in.
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
 fold=h['fold'].to_numpy();fam=h['behavior_family'].to_numpy()
 pair=h['pair_id'].to_numpy();ev=h['ev'].to_numpy();lab=h['label'].to_numpy()
 sel=np.isin(fold,[1,2,3,4])&(lab==1)
 rank=lambda x:(np.argsort(np.argsort(x))+1)/len(x)
 groups={}
 for i in np.flatnonzero(sel):groups.setdefault(pair[i],[]).append(i)
 groups={k:np.array(v) for k,v in groups.items()}
 base=np.load('artifacts/context_cv/baseline.npy');ctx=np.load('artifacts/context_cv/context.npy')
 sb={fi:np.load(f'artifacts/context_cv/seeds_base_{fi}.npy') for fi in range(3)}
 sc={}
 for fi in (0,2):
  p=Path(f'artifacts/context_cv/seeds_ctx_{fi}.npy')
  if p.exists():sc[fi]=np.load(p)
 def run(weights,use_seeds):
  per={};vals_all=[]
  for fi,f in enumerate(FAM):
   ids=[p for p,idx in groups.items() if fam[idx[0]]==f];w=weights[fi];vals=[]
   for p in ids:
    idx=groups[p]
    if use_seeds:
     rb=np.mean([rank(sb[fi][s][idx]) for s in range(sb[fi].shape[0])],0)
     rc=np.mean([rank(sc[fi][s][idx]) for s in range(sc[fi].shape[0])],0) if (w>0 and fi in sc) else None
    else:
     rb=rank(base[fi][idx]);rc=rank(ctx[fi][idx]) if w>0 else None
    s=rb if rc is None else (1-w)*rb+w*rc
    a=ap5(ev[idx][np.argsort(-s)])
    if a is not None:vals.append(a);vals_all.append(a)
   per[f]=float(np.mean(vals))
  return per,float(np.mean(vals_all))
 out={}
 for tag,ws,seeds in [('v4 weights, 1 seed',[.5,0,.5],False),
                      ('v4 weights, 3 seeds',[.5,0,.5],True),
                      ('tuned weights, 1 seed',[.7,0,.5],False),
                      ('tuned weights, 3 seeds',[.7,0,.5],True)]:
  if seeds and (0 not in sc or 2 not in sc):continue
  per,tot=run(ws,seeds);out[tag]={'overall':tot,**per}
  print(f'{tag:26} overall {tot:.4f}   '+'  '.join(f'{k[:9]} {v:.4f}' for k,v in per.items()),flush=True)
 Path('reports/seed_eval.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
