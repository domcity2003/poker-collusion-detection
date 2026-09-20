"""Action-local card context, immediate yielding, and consistent pair roles.
All features use supplied cards/actions, never labels or evidence ranks.
"""
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations
import argparse,time
import polars as pl
from features import pre,R,ST,AGG
from treys import Card,Evaluator
EV=Evaluator();CARD={r+s:Card.new(r+s) for r in R for s in 'shdc'}
STREET=['pre','flop','turn','river']
KEYS=['yield_immediate','yield_cheap','yield_junk','yield_weak_raiser','yield_price_min','joint_weak_entry','follow_weak','folded_winner','loser_pre','winner_pre','paid_weak','won_weak','signed_fold','signed_transfer','signed_raise','pre_cards_min','pre_cards_max']
for st in STREET:
 for k in ['partner_fold','partner_call','partner_raise','check','weak_aggr','price_fold','price_call','size_aggr','fold_private_pair','fold_top_pair','fold_overpair','fold_draw','call_air','check_private_pair','fold_behind','fold_ahead','act_partner','raise_weak','raise_then_yield']:
  KEYS.append(st+'_'+k)

def made(cards,board):
 if not board:return 0,0,0,0
 br=[R[x[0]] for x in board];rr=[R[x[0]] for x in cards];ss=Counter(x[1] for x in cards+board)
 private=int(any(r in br for r in rr) or rr[0]==rr[1]);top=int(max(br) in rr);over=int(rr[0]==rr[1] and rr[0]>max(br))
 ranks=set(rr+br)
 if 14 in ranks:ranks.add(1)
 draw=int(len(board)<5 and (max(ss.values())==4 or max(sum(r in ranks for r in range(i,i+5)) for i in range(1,11))>=4))
 return private,top,over,draw

def extract(pool,pairs,hands):
 s=pl.read_parquet(f'artifacts/pools/{pool}/seats.parquet');a=pl.read_parquet(f'artifacts/pools/{pool}/actions.parquet').sort('hid','action_no')
 ss=defaultdict(dict);aa=defaultdict(list)
 for r in s.iter_rows(named=True):ss[r['hid']][r['player_id']]=r
 for r in a.iter_rows(named=True):aa[r['hid']].append(r)
 lookup={tuple(sorted((r['player_1'],r['player_2']))):r['pair_id'] for r in pairs.iter_rows(named=True)}
 out=[]
 for h in hands.iter_rows(named=True):
  seat=ss[h['hid']];cand=[(lookup[k],*k) for k in combinations(sorted(seat),2) if k in lookup]
  if not cand:continue
  board=h['board_cards'].split();bb=h['big_blind'];ac=aa[h['hid']];cache={}
  def cs(p,st):
   key=(p,st)
   if key not in cache:
    x=seat[p];cards=[x['hole_card_1'],x['hole_card_2']];b=board[:[0,3,4,5][st]]
    rk=EV.evaluate([CARD[z] for z in b],[CARD[z] for z in cards]) if len(b)>=3 else -pre(*cards)
    cache[key]=(rk,*made(cards,b))
   return cache[key]
  for pid,p,q in cand:
   f=dict.fromkeys(KEYS,0.);f['yield_price_min']=1.
   sp,sq=seat[p],seat[q];pc=pre(sp['hole_card_1'],sp['hole_card_2']);qc=pre(sq['hole_card_1'],sq['hole_card_2'])
   f['pre_cards_min']=min(pc,qc);f['pre_cards_max']=max(pc,qc)
   if sp['net_chips']>sq['net_chips']:wp,lp=p,q;wpre,lpre=pc,qc
   else:wp,lp=q,p;wpre,lpre=qc,pc
   f['loser_pre']=lpre;f['winner_pre']=wpre;f['folded_winner']=float(seat[lp]['folded'] and seat[wp]['won_share']>0)
   f['paid_weak']=max(0,-seat[lp]['net_chips'])/bb*(1-lpre);f['won_weak']=max(0,seat[wp]['net_chips'])/bb*(1-wpre)
   f['signed_transfer']=(sp['net_chips']-sq['net_chips'])/bb
   live=set(seat);last={};entered=set();raised=set();previous=None
   for x in ac:
    st=ST[x['street']];z=STREET[st];actor=x['player_id'];act=x['action'];agg=act in AGG and (act!='all_in' or x['amount']>x['to_call'])
    if actor in (p,q):
     partner=q if actor==p else p;direction=1 if actor==p else -1
     if st==0 and act in ('call','raise','bet','all_in') and actor not in entered:
      if partner in entered:f['follow_weak']+=1-(pc if actor==p else qc)
      entered.add(actor)
     if partner in live:
      own=cs(actor,st);other=cs(partner,st);f[z+'_act_partner']+=1
      price=x['to_call']/max(1,x['pot_before']);weak=1-(pc if actor==p else qc)
      if st>0:weak=float(own[1]==0)
      if agg:
       f[z+'_weak_aggr']+=weak;f[z+'_size_aggr']+=x['amount']/max(1,x['pot_before']);f[z+'_raise_weak']+=weak
       f['signed_raise']+=direction
      if act=='check':f[z+'_check']+=1;f[z+'_check_private_pair']+=own[1]
      if last.get(st)==partner and x['to_call']>0:
       if act in ('fold','call','raise'):f[z+'_partner_'+act]+=1
       if act=='fold':
        f['signed_fold']+=direction;f[z+'_price_fold']+=price;f[z+'_fold_private_pair']+=own[1];f[z+'_fold_top_pair']+=own[2];f[z+'_fold_overpair']+=own[3];f[z+'_fold_draw']+=own[4]
        f[z+'_fold_ahead']+=float(own[0]<other[0]);f[z+'_fold_behind']+=float(own[0]>other[0]);f[z+'_raise_then_yield']+=actor in raised
        if previous is not None and previous['player_id']==partner and previous['action'] in AGG:
         f['yield_immediate']+=1;f['yield_cheap']+=1-min(1,price);f['yield_junk']+=weak;f['yield_weak_raiser']+=1-(qc if partner==q else pc)
         f['yield_price_min']=min(f['yield_price_min'],price)
       if act=='call':f[z+'_price_call']+=price;f[z+'_call_air']+=float(own[1]==0)
    if agg:last[st]=actor;raised.add(actor)
    if act=='fold':live.discard(actor)
    previous=x
   if p in entered and q in entered:f['joint_weak_entry']=(1-pc)*(1-qc)
   out.append({'pair_id':pid,'hand_id':h['hand_id'],**{'cx_'+k:v for k,v in f.items()}})
 df=pl.DataFrame(out).with_columns(pl.exclude('pair_id','hand_id').cast(pl.Float32))
 # Orient role-consistency by the other hands in this pair; reversing IDs leaves products invariant.
 extra=[]
 for col in ['cx_signed_transfer','cx_signed_fold','cx_signed_raise']:
  mean_other=(pl.col(col).sum().over('pair_id')-pl.col(col))/(pl.len().over('pair_id')-1).clip(1,None)
  extra.append((pl.col(col)*mean_other.sign()).alias(col.replace('signed','aligned')))
 return df.with_columns(extra).drop('cx_signed_transfer','cx_signed_fold','cx_signed_raise')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',default='development');args=ap.parse_args()
 phase=args.phase;h=pl.read_parquet('artifacts/hands.parquet').filter(pl.col('phase')==phase)
 p=pl.read_parquet('artifacts/labels.parquet') if phase=='development' else pl.read_csv('data/evaluation_pairs.csv').join(pl.read_parquet('artifacts/player_pools.parquet'),left_on='player_1',right_on='player_id')
 out=Path(f'artifacts/context_{phase}');out.mkdir(exist_ok=True);t=time.time()
 for i,pool in enumerate(sorted(p['pool'].unique())):
  file=out/f'{pool}.parquet'
  if file.exists():continue
  extract(pool,p.filter(pl.col('pool')==pool),h.filter(pl.col('pool')==pool)).write_parquet(file)
  if i%50==0:print(phase,i,round(time.time()-t),flush=True)
 print('DONE',round(time.time()-t),flush=True)
