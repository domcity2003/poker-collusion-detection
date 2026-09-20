from pathlib import Path
import hashlib,json
import polars as pl

OUT=Path('artifacts'); OUT.mkdir(exist_ok=True)
h=pl.read_parquet('data/hands.parquet').with_row_index('hid')
tables=sorted(h['table_id'].unique().to_list())
tm={t:i for i,t in enumerate(tables)}
h=h.with_columns(pl.col('table_id').replace_strict(tm).cast(pl.Int16).alias('pool'))
h.write_parquet(OUT/'hands.parquet')
print('hands',h.shape,flush=True)
s=pl.read_parquet('data/seats.parquet').join(h.select('hand_id','hid','pool'),on='hand_id').drop('hand_id')
p=s.select('player_id','pool').unique()
assert p['player_id'].n_unique()==p.height
p.write_parquet(OUT/'player_pools.parquet')
labels=pl.read_csv('data/development_labels.csv').join(p,left_on='player_1',right_on='player_id')
# Fixed assignment independent of labels. Pool zero fold is an untouched lockbox.
fm={i:int(hashlib.sha256(t.encode()).hexdigest()[:8],16)%5 for t,i in tm.items()}
labels=labels.with_columns(pl.col('pool').replace_strict(fm).alias('fold'))
labels.write_parquet(OUT/'labels.parquet')
print(labels.group_by('fold','behavior_family').len().sort('fold','behavior_family'),flush=True)
audit={'rows':{'hands':h.height,'seats':s.height},'pools':len(tables),'players':p.height,
 'seat_duplicate_keys':s.height-s.select('hid','player_id').unique().height,
 'hand_duplicate_ids':h.height-h['hand_id'].n_unique(),
 'phase':h.group_by('phase').len().to_dicts(),
 'card_nulls':s.select(pl.col('hole_card_1').null_count(),pl.col('hole_card_2').null_count()).to_dicts(),
 'chip_balance':s.group_by('hid').agg(pl.col('net_chips').sum().alias('balance')).select((pl.col('balance')!=0).sum()).to_dicts()}
for key,frame in s.partition_by('pool',as_dict=True).items():
 d=OUT/'pools'/str(key[0]);d.mkdir(parents=True,exist_ok=True);frame.write_parquet(d/'seats.parquet')
del s
print('seats partitioned',flush=True)
a=pl.read_parquet('data/actions.parquet').join(h.select('hand_id','hid','pool'),on='hand_id').drop('hand_id')
audit['rows']['actions']=a.height
audit['actions']=a.group_by('action').len().to_dicts()
audit['streets']=a.group_by('street').len().to_dicts()
for key,frame in a.partition_by('pool',as_dict=True).items():frame.write_parquet(OUT/'pools'/str(key[0])/'actions.parquet')
del a
Path('reports/data_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2),flush=True)
# Training-only case selection; full logs retained locally for review.
ev=pl.read_csv('data/development_evidence.csv').join(labels.select('pair_id','player_1','player_2','fold'),on='pair_id').filter(pl.col('fold')!=0)
chosen=pl.concat([ev.filter(pl.col('behavior_family')==b).sort('pair_id','evidence_rank').unique('pair_id',maintain_order=True).head(10) for b in ['directed_transfer','soft_play','coordinated_isolation']])
chosen.write_csv('reports/discovery_cases.csv')
ids=chosen['hand_id'].to_list()
ss=pl.read_parquet('data/seats.parquet').filter(pl.col('hand_id').is_in(ids))
aa=pl.read_parquet('data/actions.parquet').filter(pl.col('hand_id').is_in(ids))
with open('reports/discovery_replays.txt','w') as f:
 for row in chosen.iter_rows(named=True):
  f.write('\n'+str(row)+'\n'+str(h.filter(pl.col('hand_id')==row['hand_id']).to_dicts())+'\n')
  f.write('SEATS '+str(ss.filter(pl.col('hand_id')==row['hand_id']).to_dicts())+'\n')
  f.write('ACTIONS '+str(aa.filter(pl.col('hand_id')==row['hand_id']).sort('action_no').to_dicts())+'\n')
print('DONE',flush=True)
