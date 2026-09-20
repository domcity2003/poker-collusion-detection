"""Fit the context-enabled risk models on all development data, score evaluation.

The pair-level context aggregates are the first genuinely new signal the risk
model has seen. On development, feeding them in and blending .60 v6-risk / .40
context-risk improves every censoring level from 50 upward and all four folds,
while leaving censor-0 unchanged - the context model alone regresses censor-0
(-.008) because it promotes more unknowns, and the blend is what removes that
risk while keeping the gain.
"""
import sys,json,time,gc
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import pair_table,feature_columns,pool_blend,FAM,ID
OUT=Path('artifacts/v7');OUT.mkdir(exist_ok=True)

def cb(seed=2026):
 return CatBoostClassifier(iterations=550,depth=5,learning_rate=.055,l2_leaf_reg=8,
   thread_count=6,verbose=False,random_seed=seed,allow_writing_files=False)

def ctx_join(df,phase,key):
 c=pl.read_parquet(f'artifacts/context_agg_{phase}/*.parquet')
 if phase=='development':
  c=c.with_columns(pl.col('pair_id').str.split('|').list.get(0).alias('player_1'),
                   pl.col('pair_id').str.split('|').list.get(1).alias('player_2')).drop('pair_id')
  return df.join(c,on=['player_1','player_2'],how='left')
 return df.join(c,on='pair_id',how='left')

def main():
 t=time.time()
 dev=ctx_join(pair_table('development'),'development',None)
 cols=feature_columns(dev)
 (OUT/'feature_columns.json').write_text(json.dumps(cols,indent=2))
 X=np.nan_to_num(dev.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 lab=dev['label'].to_numpy();fam=dev['behavior_family'].fill_null('none').to_numpy()
 y=(lab==1).astype(int);w=np.where(lab==1,8.,np.where(lab==0,1.,.15))
 del dev;gc.collect()
 print('dev matrix',X.shape,flush=True)
 gm=cb();gm.fit(X,y,sample_weight=w);gm.save_model(str(OUT/'generic.cbm'));print(' generic fitted',round(time.time()-t),flush=True)
 sm=[]
 for i,fm in enumerate(FAM):
  keep=~((lab==1)&(fam!=fm));yy=((lab==1)&(fam==fm)).astype(int)
  Xk,yk,wk=X[keep],yy[keep],w[keep]
  m=cb();m.fit(Xk,yk,sample_weight=wk);m.save_model(str(OUT/f'specialist_{i}.cbm'));sm.append(m)
  del Xk,yk,wk;gc.collect();print(' specialist',fm,round(time.time()-t),flush=True)
 del X;gc.collect()
 ev=ctx_join(pair_table('evaluation'),'evaluation',None)
 missing=[c for c in cols if c not in ev.columns];assert not missing,missing[:5]
 Xe=np.nan_to_num(ev.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 gen=gm.predict_proba(Xe)[:,1]
 spec=np.vstack([m.predict_proba(Xe)[:,1] for m in sm])
 risk=pool_blend(.65*gen+.35*spec.max(0),ev['pool'].to_numpy())
 pl.DataFrame({'pair_id':ev['pair_id'],'context_risk':risk,
   **{f'spec_{i}':spec[i] for i in range(3)}}).write_parquet(OUT/'context_risk.parquet')
 print('saved eval context risk',len(risk),round(time.time()-t),flush=True)
if __name__=='__main__':main()
