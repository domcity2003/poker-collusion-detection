"""Pair risk experiments scored against the full candidate population.

The labelled-pairs-only CV is saturated, so every variant here is judged by
average precision over all in-fold candidates (unlabelled pairs treated as
negatives — pessimistic, but the only honest lower bound available).
  pop     - current model: cheap population features only
  rich    - population features plus the full pair-hand aggregates
  richonly- pair-hand aggregates alone, to attribute the gain
"""
import sys,json,argparse,time
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import _average_precision
ID={'pair_id','player_1','player_2','pool','fold','label','label_status','behavior_family','n_hands','hand_id','hid'}

def load():
 labels=pl.read_parquet('artifacts/labels.parquet')
 pop=pl.read_parquet('artifacts/population/development_*.parquet')
 pop=pop.filter(pl.col('n_hands')>=57).with_columns(pl.col('pool').cast(pl.Int32))
 broad=pl.read_parquet('artifacts/development_broad/*.agg.parquet')
 broad=broad.with_columns(pl.col('pair_id').str.split('|').list.get(0).alias('player_1'),
                          pl.col('pair_id').str.split('|').list.get(1).alias('player_2')).drop('pair_id','pool')
 df=pop.join(broad,on=['player_1','player_2'],how='inner')
 df=df.join(labels.select('player_1','player_2','label','behavior_family'),on=['player_1','player_2'],how='left')
 return df.with_columns(pl.col('label').fill_null(-1))

def cv(df,cols,weights_unknown=.04,it=550,depth=5,tag=''):
 X=np.nan_to_num(df.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 lab=df['label'].to_numpy();y=(lab==1).astype(int);fold=df['fold'].to_numpy()
 w=np.where(lab==1,8.,np.where(lab==0,1.,weights_unknown))
 oof=np.zeros(len(y));t=time.time()
 for f in [1,2,3,4]:
  tr=(fold!=f)&(fold!=0);te=fold==f
  m=CatBoostClassifier(iterations=it,depth=depth,learning_rate=.055,l2_leaf_reg=8,thread_count=6,verbose=False,random_seed=2026,allow_writing_files=False)
  m.fit(X[tr],y[tr],sample_weight=w[tr]);oof[te]=m.predict_proba(X[te])[:,1]
 sel=np.isin(fold,[1,2,3,4])
 known=sel&(lab>=0)
 out={'variant':tag,'n_features':len(cols),'broad_ap':_average_precision(y[sel],oof[sel]),
      'labeled_ap':_average_precision(y[known],oof[known]),
      'per_fold':[_average_precision(y[fold==f],oof[fold==f]) for f in [1,2,3,4]],
      'seconds':round(time.time()-t)}
 return out,oof

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variants',default='pop,rich,richonly');args=ap.parse_args()
 df=load();print('candidates',df.shape,'positives',int((df['label']==1).sum()),flush=True)
 popcols=[c for c,t in df.schema.items() if c not in ID and t.is_numeric() and (c.startswith('pop_') or c.startswith('contrast_') or c in('directional_net',))]
 allcols=[c for c,t in df.schema.items() if c not in ID and t.is_numeric()]
 richcols=[c for c in allcols if c not in popcols]
 sets={'pop':popcols,'rich':allcols,'richonly':richcols}
 res={}
 for v in args.variants.split(','):
  r,oof=cv(df,sets[v],tag=v);res[v]=r;print(json.dumps(r),flush=True)
  if v=='rich':
   df.select('player_1','player_2','pool','fold','label').with_columns(pl.Series('risk',oof)).write_parquet('artifacts/risk_oof_rich.parquet')
 Path('reports/risk_experiments.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
