"""Combined risk model: ratio features + family specialists + generic blend."""
import sys,json,time,argparse
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import pair_table,feature_columns,pool_blend,FAM
from ratios import add_ratios
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import _average_precision

def cb(seed=2026,it=550,depth=5):
 return CatBoostClassifier(iterations=it,depth=depth,learning_rate=.055,l2_leaf_reg=8,
   thread_count=7,verbose=False,random_seed=seed,allow_writing_files=False)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ratios',action='store_true');args=ap.parse_args()
 df=pair_table('development')
 if args.ratios:df=add_ratios(df)
 cols=feature_columns(df)
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
   keep=tr&~((lab==1)&(fam!=fm));yy=((lab==1)&(fam==fm)).astype(int)
   m=cb();m.fit(X[keep],yy[keep],sample_weight=w[keep]);spec[i,te]=m.predict_proba(X[te])[:,1]
 tag='ratios' if args.ratios else 'noratios'
 res={'tag':tag,'n_features':len(cols),'seconds':round(time.time()-t)}
 def ev(name,s):
  b=pool_blend(s,pool);res[name]=_average_precision(y[sel],b[sel]);return res[name]
 ev('generic',gen)
 for a in [.25,.35,.45]:ev(f'blend_{a}',(1-a)*gen+a*spec.max(0))
 print(json.dumps(res),flush=True)
 np.save(f'artifacts/risk_gen_{tag}.npy',gen);np.save(f'artifacts/risk_spec_{tag}.npy',spec)
 Path(f'reports/risk_full_{tag}.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
