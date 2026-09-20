"""Hindsight features: what the cards actually were, not what the pair did.

Every seat's hole cards are in the data, folded players included. The existing
feature set never uses that beyond comparing the two partners to each other, so
it cannot see the defining shape of soft play and chip dumping: a player folding
or checking down while actually holding the best hand at the table.

These are deliberately *not* decision features - they use information no player
had at the time. That is fine here: the task is detection after the fact, and it
is almost certainly how the organisers chose their evidence hands.

Written per pool keyed by (pair_id, hand_id) so it joins onto the existing
feature tables without re-extracting them.
"""
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import argparse,time
import numpy as np, polars as pl
from treys import Card,Evaluator
EV=Evaluator(); CARDS={r+s:Card.new(r+s) for r in '23456789TJQKA' for s in 'shdc'}
ST={'preflop':0,'flop':1,'turn':2,'river':3}
AGG={'raise','bet','all_in'}
CUT=(0,3,4,5)
# Fixed schema: a pool where some event never occurs must still emit the column,
# otherwise the per-pool parquets cannot be globbed back together.
SCHEMA=['p_would_win','q_would_win','pair_would_win','pair_holds_best','best_hand_folded',
 'best_folded_pot_bb','winner_is_partner_of_folder','pair_won_despite_other_best','partner_paid_off',
 'acts_with_best','fold_with_best','fold_with_best_pot_bb','fold_with_best_cheap',
 'fold_beating_partner','fold_beating_partner_pot_bb','passive_with_best','passive_with_best_pot_bb',
 'passive_beating_partner','aggressive_without_best','aggressive_behind_partner','aggressive_worst',
 'fold_with_best_any','passive_with_best_any','fold_beating_partner_any','acts_with_best_any',
 # Coordinated isolation has no chip-dump shape at all - nobody folds the best
 # hand. It is defined by what happens to the THIRD player, so it needs the
 # victim described: who was squeezed out, who stayed, and who paid the pair.
 'n_third','third_folded_pre','pair_both_agg_pre','squeeze_pre','isolated_pot',
 'survivors_after_pre','victim_loss_bb','pair_gain_bb','gain_from_victim',
 'victim_contrib_bb','victim_folded','victim_went_showdown','pair_vs_one',
 'victim_worst_hand','victim_best_hand','pair_agg_after_squeeze']

def ranks_at(seat,board,k,players):
 """treys rank (lower is better) for each player on the first k board cards."""
 if k<3:return None
 b=[CARDS[c] for c in board[:k]]
 return {p:EV.evaluate(b,[CARDS[seat[p]['hole_card_1']],CARDS[seat[p]['hole_card_2']]]) for p in players}

def pool_hindsight(pool,pairs,hands):
 ss=pl.read_parquet(f'artifacts/pools/{pool}/seats.parquet')
 aa=pl.read_parquet(f'artifacts/pools/{pool}/actions.parquet').sort('hid','action_no')
 bys=defaultdict(list);bya=defaultdict(list)
 for r in ss.iter_rows(named=True):bys[r['hid']].append(r)
 for r in aa.iter_rows(named=True):bya[r['hid']].append(r)
 lookup={tuple(sorted((r['player_1'],r['player_2']))):r['pair_id'] for r in pairs.iter_rows(named=True)}
 out=[]
 for hh in hands.iter_rows(named=True):
  hid=hh['hid'];seats=bys[hid];seat={r['player_id']:r for r in seats}
  cand=[(lookup[k],*k) for k in combinations(sorted(seat),2) if k in lookup]
  if not cand:continue
  board=hh['board_cards'].split();bb=hh['big_blind'];acts=bya[hid];allp=list(seat)
  nb=len(board)
  # Ranks on the full board that was actually dealt, and per street.
  byst={st:ranks_at(seat,board,CUT[st],allp) for st in range(4) if CUT[st]<=nb}
  final=byst.get(max([s for s in byst],default=-1))
  best_final=min(final.values()) if final else None
  winners={p for p,v in final.items() if v==best_final} if final else set()
  live=set(seat);state=[]
  for a in acts:
   st=ST[a['street']]
   state.append((a,st,live.copy()))
   if a['action']=='fold':live.discard(a['player_id'])
  for pid,p,q in cand:
   f=defaultdict(float);sp,sq=seat[p],seat[q]
   if final:
    f['p_would_win']=float(p in winners);f['q_would_win']=float(q in winners)
    f['pair_would_win']=float(bool(winners&{p,q}))
    f['pair_holds_best']=float(min(final[p],final[q])==best_final)
    # A partner surrendering the best hand is the core dump/soft-play shape.
    f['best_hand_folded']=float((p in winners and sp['folded'])or(q in winners and sq['folded']))
    f['best_folded_pot_bb']=f['best_hand_folded']*hh['final_pot']/bb
    f['winner_is_partner_of_folder']=float((p in winners and sq['folded'] and not sp['folded'])
                                          or(q in winners and sp['folded'] and not sq['folded']))
    # Did the pair's chips end up with the partner rather than the rightful winner?
    other_best=float(bool(winners-{p,q}))
    f['pair_won_despite_other_best']=other_best*float(max(sp['won_share'],sq['won_share'])>0)
    f['partner_paid_off']=float(sp['net_chips']*sq['net_chips']<0)
   # --- victim / isolation block ---
   thirds=[x for x in allp if x not in (p,q)]
   f['n_third']=len(thirds)
   pre=[a for a,st,_ in state if st==0]
   agg_pre={x['player_id'] for x in pre if x['action'] in AGG}
   f['pair_both_agg_pre']=float(p in agg_pre and q in agg_pre)
   order=[x['player_id'] for x in pre if x['action'] in AGG]
   f['squeeze_pre']=float(any(order[i] in (p,q) and order[i+1] in (p,q) and order[i]!=order[i+1]
                              for i in range(len(order)-1)))
   folded_pre={x['player_id'] for x in pre if x['action']=='fold'}
   f['third_folded_pre']=float(len(folded_pre&set(thirds)))
   survivors=[x for x in thirds if x not in folded_pre]
   f['survivors_after_pre']=float(len(survivors))
   f['pair_vs_one']=float(len(survivors)==1 and not(sp['folded'] and sq['folded']))
   f['isolated_pot']=f['pair_both_agg_pre']*float(len(survivors)<=1)
   f['pair_gain_bb']=(sp['net_chips']+sq['net_chips'])/bb
   if thirds:
    v=min(thirds,key=lambda x:seat[x]['net_chips'])
    sv=seat[v]
    f['victim_loss_bb']=-sv['net_chips']/bb
    f['gain_from_victim']=min(max(0,f['pair_gain_bb']),max(0,f['victim_loss_bb']))
    f['victim_contrib_bb']=sv['total_contribution']/bb
    f['victim_folded']=float(sv['folded']);f['victim_went_showdown']=float(sv['went_to_showdown'])
    if final and v in final:
     f['victim_worst_hand']=float(final[v]==max(final.values()))
     f['victim_best_hand']=float(final[v]==best_final)
   f['pair_agg_after_squeeze']=f['squeeze_pre']*float(sum(
     1 for a,st,_ in state if st>0 and a['player_id'] in (p,q) and a['action'] in AGG))
   for a,st,active in state:
    actor=a['player_id']
    if actor not in (p,q):continue
    partner=q if actor==p else p
    r=byst.get(st)
    if r is None or partner not in active:continue
    liv=[x for x in active if x in r]
    if len(liv)<2:continue
    mine=r[actor];best=min(r[x] for x in liv)
    am_best=float(mine==best)
    better_than_partner=float(mine<r[partner])
    price=a['to_call']/max(1,a['pot_before'])
    act=a['action']
    f['acts_with_best']+=am_best
    if act=='fold':
     # Folding the best hand at the table, and how much was surrendered.
     f['fold_with_best']+=am_best
     f['fold_with_best_pot_bb']+=am_best*a['pot_before']/bb
     f['fold_with_best_cheap']+=am_best*(1-min(1,price))
     f['fold_beating_partner']+=better_than_partner
     f['fold_beating_partner_pot_bb']+=better_than_partner*a['pot_before']/bb
    elif act in ('check','call'):
     f['passive_with_best']+=am_best
     f['passive_with_best_pot_bb']+=am_best*a['pot_before']/bb
     f['passive_beating_partner']+=better_than_partner
    elif act in AGG:
     f['aggressive_without_best']+=1-am_best
     f['aggressive_behind_partner']+=1-better_than_partner
     f['aggressive_worst']+=float(mine==max(r[x] for x in liv))
   for k in ['fold_with_best','passive_with_best','fold_beating_partner','acts_with_best']:
    f[k+'_any']=float(f[k]>0)
   out.append({'pair_id':pid,'hand_id':hh['hand_id'],'hid':hid,**{k:f[k] for k in SCHEMA}})
 if not out:return pl.DataFrame()
 return pl.DataFrame(out,infer_schema_length=None).with_columns(
   pl.exclude('pair_id','hand_id','hid').cast(pl.Float32))

def aggregate(df):
 fs=[c for c in df.columns if c not in ('pair_id','hand_id','hid')]
 ex=[]
 for f in fs:
  ex.extend([pl.col(f).mean().alias('hs_'+f+'__mean'),pl.col(f).max().alias('hs_'+f+'__max'),
             pl.col(f).top_k(5).mean().alias('hs_'+f+'__top5'),pl.col(f).sum().alias('hs_'+f+'__sum')])
 return df.group_by('pair_id').agg(ex).fill_null(0)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',default='development')
 ap.add_argument('--broad',action='store_true');ap.add_argument('--start',type=int,default=0)
 ap.add_argument('--limit',type=int,default=400);args=ap.parse_args()
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parent))
 from features import broad_pairs
 h=pl.read_parquet('artifacts/hands.parquet').filter(pl.col('phase')==args.phase)
 if args.broad:pairs=broad_pairs(args.phase)
 elif args.phase=='development':pairs=pl.read_parquet('artifacts/labels.parquet')
 else:pairs=pl.read_csv('data/evaluation_pairs.csv').join(pl.read_parquet('artifacts/player_pools.parquet'),left_on='player_1',right_on='player_id')
 out=Path('artifacts')/('hindsight_'+args.phase+('_broad' if args.broad else ''));out.mkdir(exist_ok=True,parents=True)
 t=time.time()
 for i,pool in enumerate(sorted(pairs['pool'].unique())):
  if i<args.start or i>=args.limit:continue
  path=out/(f'{pool}.agg.parquet' if args.broad else f'{pool}.parquet')
  if path.exists():continue
  df=pool_hindsight(pool,pairs.filter(pl.col('pool')==pool),h.filter(pl.col('pool')==pool))
  if df.is_empty():continue
  if args.broad:aggregate(df).write_parquet(path)
  else:df.write_parquet(path);aggregate(df).write_parquet(out/f'{pool}.agg.parquet')
  if i%50==0:print(args.phase,i,'rows',df.height,round(time.time()-t,1),flush=True)
 print('DONE',round(time.time()-t,1),flush=True)
