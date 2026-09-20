"""Construct the final risk-only A/B from a specified evidence baseline."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
base_path=Path(sys.argv[1]);out=Path('artifacts/v5/submission_v5.csv')
base=pd.read_csv(base_path,dtype=str);v2=pd.read_csv('artifacts/final/submission_v2_verified.csv',dtype=str).set_index('pair_id').loc[base.pair_id]
pred=pd.read_parquet('artifacts/v5/specialist_predictions.parquet').set_index('pair_id').loc[base.pair_id]
sub=base.copy();risk=.65*pd.to_numeric(v2.risk_score).to_numpy()+.35*pred.specialist_risk.to_numpy()
sub['risk_score']=np.clip(risk,0,1)
assert np.isfinite(risk).all()
keep=[c for c in base.columns if c!='risk_score'];assert sub[keep].equals(base[keep])
sub.to_csv(out,index=False)
manifest={'baseline_file':str(base_path),'baseline_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'only_risk_changed':True,'weights':{'verified_v2':.65,'full_feature_specialists':.35},'features':947,'models':'artifacts/v5/specialist_{0,1,2}.cbm','rows':len(sub)}
Path('artifacts/v5/manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
