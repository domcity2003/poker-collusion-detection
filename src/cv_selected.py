"""Cross-validate the full final configuration on a reduced feature set."""
import sys,json,time,argparse
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import pair_table,feature_columns,pool_blend,slim,FAM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import _average_precision

def cb(seed=2026,it=550,depth=5):
 return CatBoostClassifier(iterations=it,depth=depth,learning_rate=.055,l2_leaf_reg=8,
   thread_count=6,verbose=False,random_seed=seed,allow_writing_files=False)

def censored(y,s,lab,ks=(0,50,100,200,400)):
 out={}
 for k in ks:
  yy,ss=y,s
  if k:
   unk=np.flatnonzero(lab<0);drop=unk[np.argsort(-s[unk])[:k]]
   m=np.ones(len(s),bool);m[drop]=False;yy,ss=y[m],s[m]
  out[f'censor_{k}']=round(_average_precision(yy,ss),4)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--K',default='250');ap.add_argument('--spec',action='store_true')
 args=ap.parse_args()
 selm=json.loads(Path('artifacts/selected_features.json').read_text())
 cols=selm['all'] if args.K=='all' else selm['top'+args.K]
 df=slim(pair_table('development'),cols)
 cols=[c for c in feature_columns(df)]
 X=np.nan_to_num(df.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 lab=df['label'].to_numpy();fam=df['behavior_family'].fill_null('none').to_numpy()
 fold=df['fold'].to_numpy();pool=df['pool'].to_numpy()
 y=(lab==1).astype(int);sel=np.isin(fold,[1,2,3,4])
 w=np.where(lab==1,8.,np.where(lab==0,1.,.15))
 del df
 print('matrix',X.shape,f'{X.nbytes/1e6:.0f}MB',flush=True)
 t=time.time();gen=np.zeros(len(y));spec=np.zeros((3,len(y)))
 for f in [1,2,3,4]:
  tr=(fold!=f)&(fold!=0);te=fold==f
  m=cb();m.fit(X[tr],y[tr],sample_weight=w[tr]);gen[te]=m.predict_proba(X[te])[:,1]
  if args.spec:
   for i,fm in enumerate(FAM):
    keep=tr&~((lab==1)&(fam!=fm));yy=((lab==1)&(fam==fm)).astype(int)
    m=cb();m.fit(X[keep],yy[keep],sample_weight=w[keep]);spec[i,te]=m.predict_proba(X[te])[:,1]
  print(' fold',f,round(time.time()-t),flush=True)
 res={'K':args.K,'n_features':len(cols),'seconds':round(time.time()-t)}
 res['generic']=censored(y[sel],pool_blend(gen,pool)[sel],lab[sel])
 if args.spec:
  for a in [.25,.35,.45]:
   res[f'blend_{a}']=censored(y[sel],pool_blend((1-a)*gen+a*spec.max(0),pool)[sel],lab[sel])
  np.save(f'artifacts/cvsel_spec_{args.K}.npy',spec)
 np.save(f'artifacts/cvsel_gen_{args.K}.npy',gen)
 print(json.dumps(res,indent=2),flush=True)
 Path(f'reports/cv_selected_{args.K}.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
