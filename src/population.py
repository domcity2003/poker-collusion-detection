"""Broad pair population and label-free, held-out-fold entry policy features."""
from pathlib import Path
import hashlib,math,time
import polars as pl
from features import pre,R

def main():
 h=pl.read_parquet('artifacts/hands.parquet'); labels=pl.read_parquet('artifacts/labels.parquet')
 folds={r['pool']:r['fold'] for r in labels.select('pool','fold').unique().iter_rows(named=True)}
 for r in h.select('pool','table_id').unique().iter_rows(named=True):folds[r['pool']]=int(hashlib.sha256(r['table_id'].encode()).hexdigest()[:8],16)%5
 out=Path('artifacts/player_features');out.mkdir(exist_ok=True)
 t= time.time()
 for pool in range(400):
  path=out/f'{pool}.parquet'
  if path.exists():continue
  ss=pl.read_parquet(f'artifacts/pools/{pool}/seats.parquet')
  aa=pl.read_parquet(f'artifacts/pools/{pool}/actions.parquet')
  pf=aa.filter(pl.col('street')=='preflop').group_by('hid','player_id').agg(
   pl.col('action').is_in(['call','bet','raise','all_in']).any().cast(pl.Float32).alias('enter'),
   pl.col('action').is_in(['bet','raise','all_in']).any().cast(pl.Float32).alias('pfr'),
   pl.col('action_no').filter(pl.col('action').is_in(['call','bet','raise','all_in'])).min().fill_null(99).alias('entry_no'))
  stats=aa.group_by('hid','player_id').agg(pl.col('action').is_in(['bet','raise','all_in']).sum().alias('aggr'),(pl.col('action')=='call').sum().alias('calls'),(pl.col('action')=='check').sum().alias('checks'))
  s=ss.join(h.filter(pl.col('pool')==pool).select('hid','phase','button_seat','big_blind'),on='hid').join(pf,on=['hid','player_id'],how='left').join(stats,on=['hid','player_id'],how='left').fill_null(0)
  c1=s['hole_card_1'].to_list();c2=s['hole_card_2'].to_list()
  cls=[];strength=[]
  for a,b in zip(c1,c2):
   lo,hi=sorted([R[a[0]],R[b[0]]]);cls.append(f'{lo}_{hi}_{int(a[1]==b[1])}');strength.append(pre(a,b))
  s=s.with_columns(pl.Series('cls',cls),pl.Series('pre',strength,dtype=pl.Float32),((pl.col('seat_no')-pl.col('button_seat'))%6).alias('pos'),pl.lit(folds[pool]).alias('fold'))
  s=s.with_columns((pl.col('net_chips')/pl.col('big_blind')).alias('net'),(pl.col('total_contribution')/pl.col('big_blind')).alias('contrib'))
  s.select('hid','player_id','pool','phase','fold','cls','pos','pre','enter','pfr','entry_no','aggr','calls','checks','net','contrib').write_parquet(path)
  if pool%50==0:print('player features',pool,round(time.time()-t),flush=True)
 allscan=pl.scan_parquet([str(out/f'{i}.parquet') for i in range(400)])
 counts=allscan.filter(pl.col('phase')=='development').group_by('fold','cls','pos').agg(pl.len().alias('n'),pl.col('enter').sum().alias('k')).collect()
 pairout=Path('artifacts/population');pairout.mkdir(exist_ok=True)
 policies={}
 for f in range(-1,5):
  c=counts if f==-1 else counts.filter(pl.col('fold')!=f)
  c=c.group_by('cls','pos').agg(pl.col('n').sum(),pl.col('k').sum()).with_columns(((pl.col('k')+1)/(pl.col('n')+2)).alias('prob'))
  policies[f]=c.select('cls','pos','prob')
 for pool in range(400):
  for phase in ['development','evaluation']:
   path=pairout/f'{phase}_{pool}.parquet'
   if path.exists():continue
   s=pl.read_parquet(out/f'{pool}.parquet').filter(pl.col('phase')==phase).join(policies[folds[pool] if phase=='development' else -1],on=['cls','pos'],how='left').with_columns(pl.col('prob').fill_null(.5))
   s=s.with_columns((-pl.col('prob').log()*pl.col('enter')).alias('loose'),(-(1-pl.col('prob')).log()*(1-pl.col('enter'))).alias('tight'),(pl.col('enter')-pl.col('prob')).alias('resid'),((.6-pl.col('pre')).clip(0,1)*pl.col('enter')).alias('junk'))
   s.write_parquet(out/f'{phase}_{pool}.policy.parquet')
   bas=s.group_by('player_id').agg([pl.col(x).mean().alias('base_'+x) for x in ['enter','pfr','loose','junk','resid','aggr','calls','net']])
   fields=['hid','player_id','pre','enter','pfr','entry_no','aggr','calls','checks','net','contrib','loose','tight','resid','junk']
   a=s.select(fields).rename({'player_id':'player_1'});b=s.select(fields).rename({x:x+'2' for x in fields if x!='hid'})
   ph=a.join(b,on='hid').filter(pl.col('player_1')<pl.col('player_id2')).rename({'player_id2':'player_2'})
   ph=ph.with_columns((pl.col('enter')*pl.col('enter2')).alias('both_enter'),(pl.col('pfr')*pl.col('pfr2')).alias('both_raise'),
    pl.min_horizontal('loose','loose2').alias('joint_loose'),(pl.col('resid')*pl.col('resid2')).alias('joint_resid'),
    (pl.col('net')+pl.col('net2')).alias('joint_net'),(pl.col('net')-pl.col('net2')).alias('signed_net'),
    pl.min_horizontal('junk','junk2').alias('joint_junk'),
    pl.when(pl.col('entry_no')>pl.col('entry_no2')).then(pl.col('junk')).otherwise(pl.col('junk2')).mul(pl.col('enter')*pl.col('enter2')).alias('follower_junk'))
   feature=['both_enter','both_raise','joint_loose','joint_resid','joint_net','joint_junk','follower_junk']
   ex=[]
   for x in ['pre','enter','pfr','aggr','calls','checks','contrib','loose','tight','junk']:
    for agg in ['min','max']:
     name=x+'_'+agg;ex.append((pl.min_horizontal(x,x+'2') if agg=='min' else pl.max_horizontal(x,x+'2')).alias(name));feature.append(name)
   ph=ph.with_columns(ex)
   expressions=[pl.len().alias('n_hands'),pl.col('signed_net').mean().abs().alias('directional_net')]
   for x in feature:expressions.extend([pl.col(x).mean().alias('pop_'+x+'__mean'),pl.col(x).top_k(5).mean().alias('pop_'+x+'__top5'),pl.col(x).max().alias('pop_'+x+'__max')])
   # Individual conditional rates retained only to construct symmetric partner-vs-field differences.
   for x in ['enter','pfr','loose','junk','resid','aggr','calls','net']:expressions.extend([pl.col(x).mean().alias('cond_'+x),pl.col(x+'2').mean().alias('cond2_'+x)])
   pp=ph.group_by('player_1','player_2').agg(expressions).join(bas,left_on='player_1',right_on='player_id').join(bas.rename({x:x+'2' for x in bas.columns}),left_on='player_2',right_on='player_id2')
   extra=[]
   for x in ['enter','pfr','loose','junk','resid','aggr','calls','net']:
    d1=pl.col('cond_'+x)-pl.col('base_'+x);d2=pl.col('cond2_'+x)-pl.col('base_'+x+'2')
    extra.extend([pl.min_horizontal(d1,d2).alias('contrast_'+x+'_min'),pl.max_horizontal(d1,d2).alias('contrast_'+x+'_max'),(pl.col('base_'+x)+pl.col('base_'+x+'2')).alias('base_'+x+'_sum')])
   pp=pp.with_columns(*extra,pl.lit(pool).alias('pool'),pl.lit(folds[pool]).alias('fold')).drop([x for x in pp.columns if x.startswith('cond_') or x.startswith('cond2_') or x.startswith('base_')])
   pp.write_parquet(path)
  if pool%50==0:print('population',pool,round(time.time()-t),flush=True)
 print('DONE population',round(time.time()-t),flush=True)
if __name__=='__main__':main()
