"""Shared assembly of the two modelling tables, identical for both phases.

Pair table  = broad population features + pair-hand aggregates.
Hand table  = per-hand pair features + per-player entry-policy features.
Keeping one code path for development and evaluation is what stops the two
from silently drifting apart between training and inference.
"""
from pathlib import Path
import polars as pl

ID={'pair_id','player_1','player_2','pool','fold','label','label_status','behavior_family',
    'n_hands','hand_id','hid','evidence_rank','ev','shared_hands'}
FAM=['directed_transfer','soft_play','coordinated_isolation']
B=['none',*FAM,'other_coordination']

def _hindsight_pairs(phase):
 d='artifacts/hindsight_'+('development_broad' if phase=='development' else 'evaluation')
 hs=pl.read_parquet(d+'/*.agg.parquet')
 if phase=='development':
  return hs.with_columns(pl.col('pair_id').str.split('|').list.get(0).alias('player_1'),
                         pl.col('pair_id').str.split('|').list.get(1).alias('player_2')).drop('pair_id')
 return hs

def pair_table(phase,cache=True):
 """Cached by build_cache.py: the uncached path globs ~1,200 parquet files."""
 if cache:
  c=Path(f'artifacts/cache_{phase}.parquet')
  if c.exists():return pl.read_parquet(c)
 pop=pl.read_parquet(f'artifacts/population/{phase}_*.parquet').with_columns(pl.col('pool').cast(pl.Int32))
 if phase=='development':
  pop=pop.filter(pl.col('n_hands')>=57)
  agg=pl.read_parquet('artifacts/development_broad/*.agg.parquet').drop('pool')
  agg=agg.with_columns(pl.col('pair_id').str.split('|').list.get(0).alias('player_1'),
                       pl.col('pair_id').str.split('|').list.get(1).alias('player_2')).drop('pair_id')
  df=pop.join(agg,on=['player_1','player_2'],how='inner')
  df=df.join(_hindsight_pairs('development'),on=['player_1','player_2'],how='left')
  lab=pl.read_parquet('artifacts/labels.parquet').select('player_1','player_2','label','behavior_family')
  return (df.join(lab,on=['player_1','player_2'],how='left')
            .with_columns(pl.col('label').fill_null(-1),
                          (pl.col('player_1')+'|'+pl.col('player_2')).alias('pair_id')))
 agg=pl.read_parquet('artifacts/evaluation/*.agg.parquet').drop('pool')
 pairs=pl.read_csv('data/evaluation_pairs.csv')
 df=pairs.join(pop,on=['player_1','player_2'],how='left').join(agg,on='pair_id',how='left')
 return df.join(_hindsight_pairs('evaluation'),on='pair_id',how='left')

def slim(df,cols):
 """Project to the identifier columns plus a chosen feature set.

 Keeping all 947 feature columns alive in the frame AND in the numpy matrix is
 what pushes this 8GB machine into swap, so every consumer projects early."""
 keep=[c for c in df.columns if c in ID]+[c for c in cols if c in df.columns]
 return df.select(keep)

def feature_columns(df):
 return [c for c,t in df.schema.items() if c not in ID and t.is_numeric()]

def hand_table(phase,pool,pairs):
 """Per-hand rows for one pool with both players' policy features attached."""
 src=Path('artifacts')/phase/f'{pool}.parquet'
 h=pl.read_parquet(src).join(pairs.select('pair_id','player_1','player_2'),on='pair_id')
 s=pl.read_parquet(f'artifacts/player_features/{phase}_{pool}.policy.parquet')
 fs=['enter','pfr','pre','loose','tight','resid','junk','entry_no']
 h=(h.join(s.select('hid','player_id',*fs).rename({'player_id':'player_1'}),on=['hid','player_1'])
     .join(s.select('hid','player_id',*fs).rename({'player_id':'player_2',**{x:x+'2' for x in fs}}),on=['hid','player_2']))
 ex=[]
 for f in fs[:-1]:
  ex.extend([pl.min_horizontal(f,f+'2').alias('policy_'+f+'_min'),pl.max_horizontal(f,f+'2').alias('policy_'+f+'_max')])
 h=h.with_columns(*ex,(pl.col('resid')*pl.col('resid2')).alias('policy_resid_product'),
                      (pl.col('enter')*pl.col('enter2')).alias('policy_both_enter'))
 hs=pl.read_parquet(f'artifacts/hindsight_{phase}/{pool}.parquet').drop('hid')
 h=h.join(hs,on=['pair_id','hand_id'],how='left')
 return h.drop('player_1','player_2',*fs,*[x+'2' for x in fs]).fill_null(0)

def pool_blend(risk,pool,a=.10):
 """Blend raw risk with a within-pool z-score.

 Pools differ in baseline aggression and stakes, so raw scores are miscalibrated
 across the 400 pools that share one global ranking. A light blend fixes the
 calibration without the collapse that full per-pool normalisation causes
 (it would promote the top pair of every clean pool). Gain is flat over
 a in [0.05,0.30] on development, so the weight is not finely tuned.
 """
 t=pl.DataFrame({'pool':pool,'risk':risk})
 t=t.with_columns(((pl.col('risk')-pl.col('risk').mean())/(pl.col('risk').std()+1e-9)).over('pool').alias('z'))
 z=t['z'].to_numpy();zz=(z-z.min())/(z.max()-z.min()+1e-9)
 return (1-a)*risk+a*zz
