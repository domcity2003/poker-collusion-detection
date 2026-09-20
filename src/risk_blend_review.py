"""Sensitivity review using saved, row-aligned OOF risk predictions.
Unknowns are not established negatives; these AP values are diagnostics only.
"""
import json
from pathlib import Path
import numpy as np,polars as pl
from pipeline import pool_blend
from final import censored_ap
from official_metric import _average_precision
x=pl.read_parquet('artifacts/cache_development.parquet',columns=['label','fold','pool','pair_id','behavior_family'])
lab=x['label'].to_numpy();y=(lab==1).astype(int);sel=x['fold'].to_numpy()!=0;pool=x['pool'].to_numpy()
g=np.load('artifacts/risk_gen.npy');s=np.load('artifacts/risk_spec.npy')
g250=np.load('artifacts/cvsel_gen_250.npy');s250=np.load('artifacts/cvsel_spec_250.npy')
a=pool_blend(g,pool);b=pool_blend(.65*g250+.35*s250.max(0),pool);clean=pool_blend(.65*g+.35*s.max(0),pool)
# Fixed censor set across models, derived from the reference model only.
u=np.flatnonzero((lab<0)&sel);drop=u[np.argsort(-a[u])[:100]];fixed=sel.copy();fixed[drop]=False
res={}
for name,z in [('v2_reference',a),('v3_approx',b),('v2_v3_half',.5*a+.5*b),('specialists_no_feature_selection',clean)]:
 res[name]={'broad_ap':_average_precision(y[sel],z[sel]),'fixed_reference_censor100':_average_precision(y[fixed],z[fixed]),'per_fold':{str(f):_average_precision(y[x['fold'].to_numpy()==f],z[x['fold'].to_numpy()==f]) for f in [1,2,3,4]},'score_dependent_censor_sensitivity':censored_ap(y[sel],z[sel],lab[sel])}
print(json.dumps(res,indent=2));Path('reports/risk_blend_review.json').write_text(json.dumps(res,indent=2))
