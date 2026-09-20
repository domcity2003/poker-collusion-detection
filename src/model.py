from pathlib import Path
import sys,json,argparse,time
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import score as official_score,_average_precision,TARGET_BEHAVIORS,EVIDENCE_COLUMNS
B=['none',*TARGET_BEHAVIORS]
ID={'pair_id','player_1','player_2','pool','fold','label','label_status','behavior_family','hand_id','hid','evidence_rank','ev','label_right','behavior_family_right'}

def features(df):return [c for c,t in df.schema.items() if c not in ID and t.is_numeric()]
def matrix(df,cols):return np.nan_to_num(df.select(cols).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
def cat(it=450,depth=5,seed=2026):return CatBoostClassifier(iterations=it,depth=depth,learning_rate=.055,l2_leaf_reg=8,thread_count=6,verbose=False,random_seed=seed,allow_writing_files=False)
def solution(labels,evidence):
 s=labels.select('pair_id',pl.col('label').alias('risk_score'),pl.col('behavior_family').alias('predicted_behavior')).to_pandas()
 for k in range(1,6):
  d=evidence.filter(pl.col('evidence_rank')==k).select('pair_id',pl.col('hand_id').alias(f'evidence_hand_{k}')).to_pandas();s=s.merge(d,on='pair_id',how='left')
 return s.fillna('NO_EVIDENCE')
def evaluate(labels,evidence,pred):
 sol=solution(labels,evidence);a=sol.sort_values('pair_id');b=pred.set_index('pair_id').loc[a.pair_id].reset_index();risk=b.risk_score.to_numpy()
 pair=_average_precision(a.risk_score.to_numpy(),risk)
 beh=np.mean([_average_precision((a.predicted_behavior==x).to_numpy().astype(int),np.where(b.predicted_behavior==x,risk,0)) for x in TARGET_BEHAVIORS])
 total=official_score(sol,pred,'pair_id');ev=(total-.7*pair-.1*beh)/.2
 return {'score':total,'pair_ap':pair,'evidence_map5':ev,'behavior_map':float(beh)}
def add_policy(df,labels,phase):
 out=[]
 for pool in df['pool'].unique():
  dd=df.filter(pl.col('pool')==pool).join(labels.select('pair_id','player_1','player_2'),on='pair_id')
  s=pl.read_parquet(f'artifacts/player_features/{phase}_{pool}.policy.parquet')
  fs=['enter','pfr','pre','loose','tight','resid','junk','entry_no']
  dd=dd.join(s.select('hid','player_id',*fs).rename({'player_id':'player_1'}),on=['hid','player_1']).join(s.select('hid','player_id',*fs).rename({'player_id':'player_2',**{x:x+'2' for x in fs}}),on=['hid','player_2'])
  ex=[]
  for f in fs[:-1]:ex.extend([pl.min_horizontal(f,f+'2').alias('policy_'+f+'_min'),pl.max_horizontal(f,f+'2').alias('policy_'+f+'_max')])
  dd=dd.with_columns(*ex,(pl.col('resid')*pl.col('resid2')).alias('policy_resid_product'),(pl.col('enter')*pl.col('enter2')).alias('policy_both_enter'))
  out.append(dd.drop('player_1','player_2',*fs,*[x+'2' for x in fs]))
 return pl.concat(out)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');ap.add_argument('--lockbox',action='store_true');args=ap.parse_args()
 out=Path('artifacts/models');out.mkdir(exist_ok=True)
 labels=pl.read_parquet('artifacts/labels.parquet');ev=pl.read_csv('data/development_evidence.csv')
 pop=pl.read_parquet('artifacts/population/development_*.parquet').join(labels.select('player_1','player_2','pair_id','label','behavior_family'),on=['player_1','player_2'],how='left')
 pop=pop.filter((pl.col('n_hands')>=57)|pl.col('label').is_not_null()).with_columns(pl.col('label').fill_null(-1),pl.col('behavior_family').fill_null('unknown'))
 detail=pl.read_parquet('artifacts/development/*.agg.parquet').join(labels.drop('pool'),on='pair_id').sort('pair_id')
 pop_lab=pop.filter(pl.col('label')>=0).drop('player_1','player_2','pool','fold','label','behavior_family','n_hands')
 detail=detail.join(pop_lab,on='pair_id')
 if not Path('artifacts/dev_hands_policy.parquet').exists():
  paths=[p for p in Path('artifacts/development').glob('*.parquet') if '.agg.' not in p.name]
  hands=pl.read_parquet(paths)
  hands=add_policy(hands,labels,'development').join(labels.select('pair_id','label','behavior_family','fold'),on='pair_id').join(ev.select('pair_id','hand_id',pl.lit(1).alias('ev')),on=['pair_id','hand_id'],how='left').with_columns(pl.col('ev').fill_null(0))
  assert hands['ev'].sum()==ev.height,(hands['ev'].sum(),ev.height)
  hands.write_parquet('artifacts/dev_hands_policy.parquet')
 hands=pl.read_parquet('artifacts/dev_hands_policy.parquet')
 fc,pc,hc=features(detail),features(pop),features(hands)
 Path('artifacts/feature_columns.json').write_text(json.dumps({'detail':fc,'population':pc,'hand':hc},indent=2))
 dx,px,hx=matrix(detail,fc),matrix(pop,pc),matrix(hands,hc)
 dy=np.array([B.index(x) for x in detail['behavior_family']]);py=(pop['label'].to_numpy()==1).astype(int)
 hy=hands['ev'].to_numpy();dfold=detail['fold'].to_numpy();pfold=pop['fold'].to_numpy();hfold=hands['fold'].to_numpy();hl=hands['label'].to_numpy()
 print('datasets',detail.shape,pop.shape,hands.shape,'features',len(fc),len(pc),len(hc),flush=True)
 folds=[-1] if args.final else ([0] if args.lockbox else [1,2,3,4]);reports=[];predictions=[];pop_oof=[]
 for fold in folds:
  t=time.time()
  dtr=np.ones(len(dy),bool) if fold==-1 else ((dfold!=fold)&(dfold!=0));ptr=np.ones(len(py),bool) if fold==-1 else ((pfold!=fold)&(pfold!=0));htr=np.ones(len(hy),bool) if fold==-1 else ((hfold!=fold)&(hfold!=0))
  if args.lockbox:dtr=dfold!=0;ptr=pfold!=0;htr=hfold!=0
  dt=dfold==fold;pt=pfold==fold;ht=hfold==fold
  # Multiclass model on confirmed pairs for disclosed behavior and a PN risk reference.
  dm=cat(it=650);dm.fit(dx[dtr],dy[dtr]);dm.save_model(str(out/f'detail_{fold}.cbm'))
  # Broad population: unknown pairs receive weak negative weight, never confirmed labels.
  weights=np.where(pop['label'].to_numpy()==1,8.,np.where(pop['label'].to_numpy()==0,1.,.04))
  pm=cat(it=550,depth=5);pm.fit(px[ptr],py[ptr],sample_weight=weights[ptr]);pm.save_model(str(out/f'population_{fold}.cbm'))
  # Retrieval is learned within positive pairs; unlisted hands are retrieval distractors.
  hm=cat(it=500,depth=5);hr=htr&(hl==1);hm.fit(hx[hr],hy[hr]);hm.save_model(str(out/f'hand_{fold}.cbm'))
  if fold==-1:
   pl.DataFrame({'feature':pc,'importance':pm.feature_importances_}).sort('importance',descending=True).write_csv('reports/population_importance.csv')
   pl.DataFrame({'feature':hc,'importance':hm.feature_importances_}).sort('importance',descending=True).write_csv('reports/hand_importance.csv')
   print('FINAL MODELS SAVED',flush=True);continue
  dp=dm.predict_proba(dx[dt]);pp=pm.predict_proba(px[pt])[:,1];hp=hm.predict_proba(hx[ht])[:,1]
  valid=detail.filter(pl.Series(dt));hpairs=hands.filter(pl.Series(ht)).select('pair_id','hand_id','ev').with_columns(pl.Series('hand_score',hp))
  hpairs.write_parquet(out/f'hand_oof_{fold}.parquet')
  top=hpairs.sort('hand_score',descending=True).group_by('pair_id',maintain_order=True).agg(pl.col('hand_id').head(5)).to_dicts();top={r['pair_id']:r['hand_id'] for r in top}
  pv=pop.filter(pl.Series(pt)).select('pair_id','player_1','player_2','label','pool').with_columns(pl.Series('risk',pp));pv.write_parquet(out/f'population_oof_{fold}.parquet')
  rp=valid.select('pair_id').join(pv.select('pair_id',pl.col('risk').alias('risk_score')),on='pair_id').to_pandas()
  rp['pn_risk']=1-dp[:,0];rp['predicted_behavior']=np.array(B)[dp[:,1:].argmax(axis=1)+1]
  for i,b in enumerate(B):rp['p_'+b]=dp[:,i]
  for k in range(5):rp[f'evidence_hand_{k+1}']=[top.get(p,[])[k] if len(top.get(p,[]))>k else 'NO_EVIDENCE' for p in rp.pair_id]
  m=evaluate(labels.filter(pl.col('fold')==fold),ev,rp);m['fold']=fold
  alt=rp.copy();alt['risk_score']=alt.pn_risk;m['pn_reference']=evaluate(labels.filter(pl.col('fold')==fold),ev,alt)
  reports.append(m);predictions.append(rp);print('FOLD',fold,json.dumps(m),'seconds',round(time.time()-t),flush=True)
 if reports:
  pred=pd.concat(predictions);pred.to_csv('artifacts/'+('lockbox_predictions.csv' if args.lockbox else 'oof_predictions.csv'),index=False)
  combined=evaluate(labels.filter(pl.col('fold').is_in(folds)),ev,pred)
  Path('reports/'+('lockbox.json' if args.lockbox else 'validation.json')).write_text(json.dumps({'folds':reports,'pooled':combined},indent=2))
  print('POOLED',combined,flush=True)
if __name__=='__main__':main()
