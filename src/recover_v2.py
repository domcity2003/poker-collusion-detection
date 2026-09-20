"""Rebuild frozen v2 without overwriting any submitted file; persist all models."""
import importlib.machinery,importlib.util,json,gc,hashlib
from pathlib import Path
import numpy as np,polars as pl
loader=importlib.machinery.SourceFileLoader('v2','src/final_v2.py.bak')
spec=importlib.util.spec_from_loader(loader.name,loader);v=importlib.util.module_from_spec(spec);loader.exec_module(v)
out=Path('artifacts/recovered_v2');out.mkdir(exist_ok=True)
dev=v.pair_table('development');cols=v.feature_columns(dev)
r,b=v.fit_pair_models(dev,cols,np.ones(dev.height,bool));r.save_model(str(out/'risk.cbm'));b.save_model(str(out/'behavior.cbm'))
del dev;gc.collect()
ms,fc=v.fit_evidence()
for i,m in enumerate(ms):m.save_model(str(out/f'evidence_{i}.cbm'))
(out/'columns.json').write_text(json.dumps({'pair':cols,'hand':fc}))
ev=v.pair_table('evaluation');X=v.mat(ev,cols)
raw=r.predict_proba(X)[:,1];bp=b.predict_proba(X)
ev.select('pair_id','pool').with_columns(pl.Series('raw_risk',raw),*[pl.Series('family_'+str(i),bp[:,i]) for i in range(3)]).write_parquet(out/'predictions.parquet')
del X;gc.collect()
sub=v.build(ev,cols,r,b,ms,fc,'evaluation',sorted(ev['pool'].unique().to_list()))
sub.to_csv(out/'submission.csv',index=False)
print('sha256',hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(),flush=True)
