"""One risk detector per behaviour family, instead of a single generic one.

The generic model lifts directed_transfer from AP .19 (best single feature) to
.51, but coordinated_isolation only .30 -> .43. Isolation is 79 of 304 training
positives, so a model fitted to all families spends its capacity elsewhere and
under-fits the minority signature - the same asymmetry that made family
specialists work for evidence retrieval.

Caveat that shapes the design: evaluation contains an `other_coordination`
family with zero development examples. A pure max-of-specialists would have no
detector for it, so the generic model is retained and blended rather than
replaced.
"""
import sys,json,time
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import pair_table,feature_columns,pool_blend,FAM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import _average_precision

def cb(seed=2026,it=550,depth=5):
 return CatBoostClassifier(iterations=it,depth=depth,learning_rate=.055,l2_leaf_reg=8,
   thread_count=7,verbose=False,random_seed=seed,allow_writing_files=False)

def main():
 df=pair_table('development');cols=feature_columns(df)
 X=np.nan_to_num(df.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 lab=df['label'].to_numpy();fam=df['behavior_family'].fill_null('unknown').to_numpy()
 fold=df['fold'].to_numpy();pool=df['pool'].to_numpy()
 y=(lab==1).astype(int);sel=np.isin(fold,[1,2,3,4])
 w=np.where(lab==1,8.,np.where(lab==0,1.,.15))
 gen=np.zeros(len(y));spec=np.zeros((3,len(y)))
 t=time.time()
 for f in [1,2,3,4]:
  tr=(fold!=f)&(fold!=0);te=fold==f
  m=cb();m.fit(X[tr],y[tr],sample_weight=w[tr]);gen[te]=m.predict_proba(X[te])[:,1]
  for i,fm in enumerate(FAM):
   # Positives of this family only; every other pair keeps its usual weight,
   # so other families' positives are NOT treated as negatives here.
   keep=tr&~((lab==1)&(fam!=fm))
   yy=((lab==1)&(fam==fm)).astype(int)
   m=cb();m.fit(X[tr&keep],yy[tr&keep],sample_weight=w[tr&keep])
   spec[i,te]=m.predict_proba(X[te])[:,1]
 print('fitted',round(time.time()-t),'s',flush=True)
 res={}
 def ev(name,s):
  b=pool_blend(s,pool);r={'raw':_average_precision(y[sel],s[sel]),'blend':_average_precision(y[sel],b[sel])}
  for i,fm in enumerate(FAM):
   keep=sel&((lab<0)|((lab==1)&(fam==fm)))
   r[fm]=_average_precision(((lab==1)&(fam==fm))[keep].astype(int),b[keep])
  res[name]=r;print(name,json.dumps({k:round(v,4) for k,v in r.items()}),flush=True)
 ev('generic',gen)
 ev('spec_max',spec.max(0))
 ev('spec_mean',spec.mean(0))
 for a in [.2,.35,.5,.65]:
  ev(f'blend_{a}',(1-a)*gen+a*spec.max(0))
 # Rank-space blend avoids the two scores' different calibrations.
 rg=np.argsort(np.argsort(gen))/len(gen);rs=np.argsort(np.argsort(spec.max(0)))/len(gen)
 for a in [.3,.5]:ev(f'rankblend_{a}',(1-a)*rg+a*rs)
 np.save('artifacts/risk_gen.npy',gen);np.save('artifacts/risk_spec.npy',spec)
 Path('reports/risk_specialists.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
