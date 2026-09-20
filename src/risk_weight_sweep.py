"""How much should the 136k unlabelled pairs count as negatives?

The model currently separates disclosed positives from disclosed negatives
almost perfectly (AP .997) but only reaches .71 against the full candidate
population. Those are different problems. The unknown weight is the dial between
them and it has never been tested: at .04 the 136k unknowns carry total weight
~5.4k against 1.5k of curated negatives, so the curated set still dominates
training. Raising it shifts the model towards the ranking it is actually scored on.
"""
import sys,json,time
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from risk_exp import load,ID
from pipeline import pool_blend
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import _average_precision

def main():
 df=load()
 hs=pl.read_parquet('artifacts/hindsight_development_broad/*.agg.parquet')
 hs=hs.with_columns(pl.col('pair_id').str.split('|').list.get(0).alias('player_1'),
                    pl.col('pair_id').str.split('|').list.get(1).alias('player_2')).drop('pair_id')
 df=df.join(hs,on=['player_1','player_2'],how='left')
 cols=[c for c,t in df.schema.items() if c not in ID and t.is_numeric()]
 X=np.nan_to_num(df.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 lab=df['label'].to_numpy();y=(lab==1).astype(int);fold=df['fold'].to_numpy()
 pool=df['pool'].to_numpy();sel=np.isin(fold,[1,2,3,4])
 res=[]
 for pw,uw in [(8,.01),(8,.04),(8,.15),(8,.5),(8,1.),(1,1.),(20,.04),(8,.04)]:
  t=time.time();oof=np.zeros(len(y))
  w=np.where(lab==1,float(pw),np.where(lab==0,1.,float(uw)))
  for f in [1,2,3,4]:
   tr=(fold!=f)&(fold!=0);te=fold==f
   m=CatBoostClassifier(iterations=550,depth=5,learning_rate=.055,l2_leaf_reg=8,
     thread_count=7,verbose=False,random_seed=2026,allow_writing_files=False)
   m.fit(X[tr],y[tr],sample_weight=w[tr]);oof[te]=m.predict_proba(X[te])[:,1]
  bl=pool_blend(oof,pool)
  r={'pos_w':pw,'unk_w':uw,'broad_ap':_average_precision(y[sel],oof[sel]),
     'broad_ap_blend':_average_precision(y[sel],bl[sel]),'seconds':round(time.time()-t)}
  print(json.dumps(r),flush=True);res.append(r)
  np.save(f'artifacts/risk_oof_w{pw}_{uw}.npy',oof)
 Path('reports/risk_weight_sweep.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
