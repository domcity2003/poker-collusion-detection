"""Print readable hand replays so evidence hands can be eyeballed against the rest."""
import sys
import polars as pl
def replay(pool,hid,p,q,mark=''):
 s=pl.read_parquet(f'artifacts/pools/{pool}/seats.parquet').filter(pl.col('hid')==hid)
 a=pl.read_parquet(f'artifacts/pools/{pool}/actions.parquet').filter(pl.col('hid')==hid).sort('action_no')
 h=pl.read_parquet('artifacts/hands.parquet').filter(pl.col('hid')==hid).row(0,named=True)
 tag={p:'P1',q:'P2'}
 print(f"--- {mark} hand {h['hand_id']} board[{h['board_cards']}] pot={h['final_pot']} bb={h['big_blind']} showdown_n={h['players_at_showdown']}")
 for r in s.sort('seat_no').iter_rows(named=True):
  print(f"    {tag.get(r['player_id'],'  '):3} seat{r['seat_no']} {r['hole_card_1']}{r['hole_card_2']} "
        f"contrib={r['total_contribution']:>6} net={r['net_chips']:>7} "
        f"{'FOLD' if r['folded'] else ''}{' SD' if r['went_to_showdown'] else ''}{' WON' if r['won_share']>0 else ''}")
 st=None
 for r in a.iter_rows(named=True):
  if r['street']!=st:st=r['street'];print(f"    [{st}]")
  print(f"      {tag.get(r['player_id'],'  '):3} {r['action']:<7} amt={r['amount']:<7} to_call={r['to_call']:<7} pot={r['pot_before']:<7} live={r['players_active']}")
if __name__=='__main__':
 fam=sys.argv[1] if len(sys.argv)>1 else 'coordinated_isolation'
 n=int(sys.argv[2]) if len(sys.argv)>2 else 1
 h=pl.read_parquet('artifacts/dev_hands_hindsight.parquet')
 lab=pl.read_parquet('artifacts/labels.parquet')
 pairs=lab.filter(pl.col('behavior_family')==fam).head(n)
 hands=pl.read_parquet('artifacts/hands.parquet')
 for r in pairs.iter_rows(named=True):
  d=h.filter(pl.col('pair_id')==r['pair_id'])
  print(f"\n===== pair {r['pair_id']} family={fam} pool={r['pool']} hands={d.height}")
  ids=hands.select('hand_id','hid')
  ev=d.filter(pl.col('ev')==1).join(ids,on='hand_id')
  # Most "interacting" non-evidence hands, the ones a ranker confuses with evidence.
  non=(d.filter(pl.col('ev')==0).sort('facing_partner_sum',descending=True).head(3).join(ids,on='hand_id'))
  for x in ev.iter_rows(named=True):replay(r['pool'],x['hid'],r['player_1'],r['player_2'],'EVIDENCE')
  for x in non.iter_rows(named=True):replay(r['pool'],x['hid'],r['player_1'],r['player_2'],'not-listed')
