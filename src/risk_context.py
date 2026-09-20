"""Does pair-level context signal improve the RISK model (the 70% component)?

The context features were built for evidence retrieval and have never been fed
to the pair model. Judged with censored AP: development labels only 1.3% of
candidates, so uncensored broad AP charges us for ranking undisclosed colluders
highly. Censoring the most suspicious unknowns brackets the real metric (see the
handoff), so every variant is read across several censoring levels, never at
zero alone.

Generic + three family specialists, blended .65/.35 exactly as the shipped v5
risk is, so a gain here transfers to the same construction.
"""
import sys,json,time,argparse
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import pair_table,feature_columns,pool_blend,slim,FAM,ID
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
  out[f'c{k}']=round(_average_precision(yy,ss),4)
 return out

def load(with_context):
 df=pair_table('development')
 if with_context:
  ctx=pl.read_parquet('artifacts/context_agg_development/*.parquet')
  ctx=ctx.with_columns(pl.col('pair_id').str.split('|').list.get(0).alias('player_1'),
                       pl.col('pair_id').str.split('|').list.get(1).alias('player_2')).drop('pair_id')
  df=df.join(ctx,on=['player_1','player_2'],how='left')
 return df

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',action='store_true');args=ap.parse_args()
 tag='with_context' if args.context else 'baseline'
 df=load(args.context);cols=feature_columns(df)
 X=np.nan_to_num(df.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 lab=df['label'].to_numpy();fam=df['behavior_family'].fill_null('none').to_numpy()
 fold=df['fold'].to_numpy();pool=df['pool'].to_numpy()
 y=(lab==1).astype(int);sel=np.isin(fold,[1,2,3,4])
 w=np.where(lab==1,8.,np.where(lab==0,1.,.15))
 del df
 print(tag,'matrix',X.shape,f'{X.nbytes/1e6:.0f}MB',flush=True)
 gen=np.zeros(len(y));spec=np.zeros((3,len(y)));t=time.time()
 for f in [1,2,3,4]:
  tr=(fold!=f)&(fold!=0);te=fold==f
  m=cb();m.fit(X[tr],y[tr],sample_weight=w[tr]);gen[te]=m.predict_proba(X[te])[:,1]
  for i,fm in enumerate(FAM):
   keep=tr&~((lab==1)&(fam!=fm));yy=((lab==1)&(fam==fm)).astype(int)
   m=cb();m.fit(X[keep],yy[keep],sample_weight=w[keep]);spec[i,te]=m.predict_proba(X[te])[:,1]
  print(' fold',f,round(time.time()-t),flush=True)
 res={'tag':tag,'n_features':len(cols),'seconds':round(time.time()-t)}
 res['generic']=censored(y[sel],pool_blend(gen,pool)[sel],lab[sel])
 res['v5_blend_.35']=censored(y[sel],pool_blend(.65*gen+.35*spec.max(0),pool)[sel],lab[sel])
 np.save(f'artifacts/rc_gen_{tag}.npy',gen);np.save(f'artifacts/rc_spec_{tag}.npy',spec)
 print(json.dumps(res,indent=2),flush=True)
 Path(f'reports/risk_context_{tag}.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
