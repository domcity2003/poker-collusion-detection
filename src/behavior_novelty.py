"""Can we spot a behaviour family we have never seen?

Evaluation contains `other_coordination`; development contains zero examples of
it. Guessing one of the three known families for such a pair is not neutral -
it inserts a high-risk false positive into that family's average precision, so
labelling it `other_coordination` scores strictly better (verified against the
official scorer).

We cannot train on the real unseen family, but we can rehearse it: hide one of
the three known families from the classifier and ask whether low confidence
across the families it still knows identifies the hidden one. That is exactly
the decision we need to make at inference.
"""
import sys,json
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import pair_table,feature_columns,FAM

def main():
 df=pair_table('development').filter(pl.col('label')==1)
 cols=feature_columns(df)
 X=np.nan_to_num(df.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 fam=df['behavior_family'].to_numpy();fold=df['fold'].to_numpy()
 out={}
 for hidden in FAM:
  seen=[f for f in FAM if f!=hidden]
  conf=np.zeros(len(fam))
  for f in [1,2,3,4]:
   tr=(fold!=f)&(fold!=0)&(fam!=hidden);te=fold==f
   y=np.array([seen.index(x) if x in seen else -1 for x in fam])
   m=CatBoostClassifier(iterations=400,depth=4,learning_rate=.06,thread_count=7,
     verbose=False,random_seed=2026,allow_writing_files=False)
   m.fit(X[tr],y[tr]);conf[te]=m.predict_proba(X[te]).max(1)
  sel=np.isin(fold,[1,2,3,4])
  hid=(fam==hidden)&sel;kn=(fam!=hidden)&sel
  # Separation of the hidden family from the seen ones by (1 - max prob).
  from official_metric import _average_precision
  s=1-conf[sel];yy=hid[sel].astype(int)
  r={'hidden':hidden,'n_hidden':int(hid.sum()),'n_seen':int(kn.sum()),
     'novelty_ap':_average_precision(yy,s),'base_rate':float(yy.mean()),
     'median_conf_hidden':float(np.median(conf[hid])),'median_conf_seen':float(np.median(conf[kn]))}
  for t in (.5,.6,.7,.8,.9):
   flag=conf<t
   r[f'thr_{t}']={'caught_hidden':float(flag[hid].mean()),'wrongly_flagged_seen':float(flag[kn].mean())}
  out[hidden]=r;print(json.dumps(r),flush=True)
 Path('reports/behavior_novelty.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
 main()
