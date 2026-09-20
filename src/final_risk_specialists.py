"""Train full-feature specialist risk models, avoiding leaked feature selection.
Final risk: .65 verified-v2 risk + .35 pool-normalized maximum specialist.
Evidence/behavior supplied separately after the evidence-only A/B is scored.
"""
from pathlib import Path
import gc,json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from pipeline import FAM,feature_columns,pool_blend
from final import mat
out=Path('artifacts/v5');out.mkdir(exist_ok=True)
d=pl.read_parquet('artifacts/cache_development.parquet');fc=feature_columns(d);X=mat(d,fc);lab=d['label'].to_numpy();fam=d['behavior_family'].fill_null('none').to_numpy()
w=np.where(lab==1,8.,np.where(lab==0,1.,.15));del d;gc.collect();models=[]
for i,fm in enumerate(FAM):
 path=out/f'specialist_{i}.cbm';m=CatBoostClassifier(iterations=550,depth=5,learning_rate=.055,l2_leaf_reg=8,thread_count=5,verbose=False,random_seed=2026,allow_writing_files=False)
 if path.exists():m.load_model(str(path))
 else:
  keep=~((lab==1)&(fam!=fm));xx=X[keep];yy=((lab==1)&(fam==fm)).astype(int)[keep];ww=w[keep]
  m.fit(xx,yy,sample_weight=ww);m.save_model(str(path));del xx,yy,ww;gc.collect()
 models.append(m);print('fitted',fm,flush=True)
del X;gc.collect();e=pl.read_parquet('artifacts/cache_evaluation.parquet');X=mat(e,fc);pred=np.vstack([m.predict_proba(X)[:,1] for m in models]);del X;gc.collect()
s=pool_blend(pred.max(0),e['pool'].to_numpy());e.select('pair_id','pool').with_columns(pl.Series('specialist_risk',s),*[pl.Series('specialist_'+FAM[i],pred[i]) for i in range(3)]).write_parquet(out/'specialist_predictions.parquet')
(out/'feature_columns.json').write_text(json.dumps(fc));print('SAVED RISK CANDIDATE',flush=True)
