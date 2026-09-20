"""Pool- and partner-relative views of each pair.

Collusion is abnormal *relative to its own neighbourhood*: a colluding pair
should stand out among the other pairs in its 30-player pool, and above all
among the other partners each of its two players sits with. Absolute feature
values cannot express that, so for a chosen set of signal columns we add
  _pz   - z-score of the value within the pool
  _prk  - the pair's percentile within the pool
  _ptop - symmetric partner standing: min over the two players of how the pair
          ranks among that player's own partners (1.0 = this is their most
          extreme partner), which is exactly the "one special partner" shape
  _pgap - symmetric gap to the player's next-best partner
"""
import polars as pl

def signal_columns(df,k=60):
 pri=[c for c in df.columns if c.startswith('pop_') or c.startswith('contrast_')]
 return pri[:k] if k else pri

def add_relative(df,cols):
 ex=[]
 for c in cols:
  ex.append(((pl.col(c)-pl.col(c).mean())/(pl.col(c).std()+1e-9)).over('pool').alias(c+'_pz'))
  ex.append((pl.col(c).rank()/pl.len()).over('pool').alias(c+'_prk'))
 df=df.with_columns(ex)
 # Partner standing needs a long form: one row per (player, partner).
 a=df.select('player_1','player_2','pool',*cols).rename({'player_1':'p','player_2':'o'})
 b=df.select('player_1','player_2','pool',*cols).rename({'player_2':'p','player_1':'o'})
 long=pl.concat([a,b.select(a.columns)])
 ex=[]
 for c in cols:
  ex.append((pl.col(c).rank()/pl.len()).over('p').alias(c+'_r'))
  ex.append((pl.col(c)-pl.col(c).top_k(2).min()).over('p').alias(c+'_g'))
 long=long.with_columns(ex)
 keep=['p','o']+[c+s for c in cols for s in ('_r','_g')]
 l1=long.select(keep).rename({'p':'player_1','o':'player_2',**{c+s:c+s+'1' for c in cols for s in ('_r','_g')}})
 l2=long.select(keep).rename({'p':'player_2','o':'player_1',**{c+s:c+s+'2' for c in cols for s in ('_r','_g')}})
 df=df.join(l1,on=['player_1','player_2']).join(l2,on=['player_1','player_2'])
 ex=[]
 for c in cols:
  ex.append(pl.min_horizontal(c+'_r1',c+'_r2').alias(c+'_ptop'))
  ex.append(pl.min_horizontal(c+'_g1',c+'_g2').alias(c+'_pgap'))
 drop=[c+s+i for c in cols for s in ('_r','_g') for i in ('1','2')]
 return df.with_columns(ex).drop(drop)
