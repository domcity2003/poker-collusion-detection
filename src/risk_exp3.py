"""Attack the 70% component: the hidden-positive problem in training.

Development discloses only 1,860 of 137,739 candidate pairs, and the disclosed
set is a deliberate 1:4 positive:negative sample. So an unknown development pair
is NOT a negative - a meaningful number of them are colluding pairs the
organisers simply did not label. Training calls all of them negative at weight
.04, which tells the model that its most confident detections are wrong.

The evaluation solution has no such hole: every truly colluding evaluation pair
is labelled positive there. That asymmetry is also why broad AP reads ~0.68
while the leaderboard implies a real pair AP near 0.90 - broad AP charges us for
recovering exactly the hidden positives the real metric rewards.

Variants:
  base      current weighting
  spy       iterative PU relabelling - fit, then stop treating the top-scoring
            unknowns as negatives, refit. Standard PU practice.
  seeds     average several seeds (variance reduction, no label assumptions)
  tuned     more capacity
Broad AP UNDERSTATES `spy` by construction, so it is read alongside the AP over
labelled pairs only and the rank of the labelled positives.
"""
import sys,json,argparse,time
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from risk_exp import load,ID
from pipeline import pool_blend
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import _average_precision

def model(it=550,depth=5,seed=2026):
 return CatBoostClassifier(iterations=it,depth=depth,learning_rate=.055,l2_leaf_reg=8,
   thread_count=6,verbose=False,random_seed=seed,allow_writing_files=False)

def report(tag,df,oof,extra=None):
 lab=df['label'].to_numpy();y=(lab==1).astype(int);fold=df['fold'].to_numpy()
 sel=np.isin(fold,[1,2,3,4]);known=sel&(lab>=0)
 bl=pool_blend(oof,df['pool'].to_numpy())
 o=np.argsort(-bl[sel]);rk=np.empty(sel.sum(),int);rk[o]=np.arange(sel.sum())
 pr=rk[y[sel]==1]
 r={'variant':tag,'broad_ap':_average_precision(y[sel],oof[sel]),
    'broad_ap_blend':_average_precision(y[sel],bl[sel]),
    'labeled_ap_blend':_average_precision(y[known],bl[known]),
    'median_positive_rank':float(np.median(pr)),'p90_positive_rank':float(np.percentile(pr,90))}
 if extra:r.update(extra)
 print(json.dumps(r),flush=True);return r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variants',default='base,spy,seeds')
 ap.add_argument('--hindsight',action='store_true');args=ap.parse_args()
 df=load()
 if args.hindsight:
  hs=pl.read_parquet('artifacts/hindsight_development_broad/*.agg.parquet')
  hs=hs.with_columns(pl.col('pair_id').str.split('|').list.get(0).alias('player_1'),
                     pl.col('pair_id').str.split('|').list.get(1).alias('player_2')).drop('pair_id')
  df=df.join(hs,on=['player_1','player_2'],how='left')
 cols=[c for c,t in df.schema.items() if c not in ID and t.is_numeric()]
 print('table',df.shape,'features',len(cols),flush=True)
 X=np.nan_to_num(df.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 lab=df['label'].to_numpy();y=(lab==1).astype(int);fold=df['fold'].to_numpy()
 res=[]
 for v in args.variants.split(','):
  t=time.time();oof=np.zeros(len(y))
  for f in [1,2,3,4]:
   tr=(fold!=f)&(fold!=0);te=fold==f
   w=np.where(lab==1,8.,np.where(lab==0,1.,.04))
   if v=='seeds':
    ps=[]
    for sd in (2026,7,991):
     m=model(seed=sd);m.fit(X[tr],y[tr],sample_weight=w[tr]);ps.append(m.predict_proba(X[te])[:,1])
    oof[te]=np.mean(ps,0);continue
   if v=='tuned':
    m=model(it=1200,depth=7);m.fit(X[tr],y[tr],sample_weight=w[tr]);oof[te]=m.predict_proba(X[te])[:,1];continue
   m=model();m.fit(X[tr],y[tr],sample_weight=w[tr])
   if v=='spy':
    # Unknowns the first pass ranks alongside the confirmed positives are very
    # likely undisclosed positives. Stop counting them as negatives rather than
    # flipping them to positive - we have no proof, only a strong suspicion.
    s=m.predict_proba(X)[:,1];unk=(lab<0)&tr
    cut=np.quantile(s[unk],1-.02)
    w=np.where(lab==1,8.,np.where(lab==0,1.,np.where(s>cut,0.,.04)))
    m=model();m.fit(X[tr],y[tr],sample_weight=w[tr])
   oof[te]=m.predict_proba(X[te])[:,1]
  res.append(report(v,df,oof,{'seconds':round(time.time()-t)}))
  np.save(f'artifacts/risk_oof_{v}.npy',oof)
 Path('reports/risk_exp3.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
