"""Rank features once, cheaply, then never hold the full matrix again.

On an 8GB box the 137,739 x 947 float32 matrix (522MB) plus CatBoost's own
copies drives the machine into several GB of swap, and a fit that takes 18s
with free memory takes over two minutes while thrashing. Importance is
therefore computed on a row subsample (every labelled pair plus a sample of
unknowns), saved to disk, and every later step loads only the selected columns.

Selection uses training folds only; fold 0 is untouched.
"""
import sys,json,time
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import pair_table,feature_columns
def cb(seed=2026,it=400,depth=5):
 return CatBoostClassifier(iterations=it,depth=depth,learning_rate=.055,l2_leaf_reg=8,
   thread_count=6,verbose=False,random_seed=seed,allow_writing_files=False)
def main():
 df=pair_table('development');cols=feature_columns(df)
 lab=df['label'].to_numpy();fold=df['fold'].to_numpy()
 rng=np.random.default_rng(0)
 unk=np.flatnonzero((lab<0)&(fold!=0))
 keep=np.concatenate([np.flatnonzero((lab>=0)&(fold!=0)),rng.choice(unk,40000,replace=False)])
 keep.sort()
 sub=df[keep]
 X=np.nan_to_num(sub.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 y=(sub['label'].to_numpy()==1).astype(int)
 l=sub['label'].to_numpy();w=np.where(l==1,8.,np.where(l==0,1.,.15))
 del df,sub
 print('importance subsample',X.shape,'positives',int(y.sum()),flush=True)
 t=time.time();m=cb();m.fit(X,y,sample_weight=w)
 imp=m.feature_importances_;order=np.argsort(-imp)
 out={'all':cols}
 for K in (400,250,150,80):out[f'top{K}']=[cols[i] for i in order[:K]]
 out['importance']={cols[i]:float(imp[i]) for i in order[:50]}
 Path('artifacts/selected_features.json').write_text(json.dumps(out,indent=2))
 print('saved selection in',round(time.time()-t),'s. Top 15:',flush=True)
 for i in order[:15]:print(f'   {imp[i]:6.2f}  {cols[i]}',flush=True)
if __name__=='__main__':main()
