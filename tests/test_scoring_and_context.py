import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd,polars as pl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import score,ParticipantVisibleError
from context_features import extract

class ScoringContract(unittest.TestCase):
 def fixture(self):
  truth=pd.DataFrame({'pair_id':['A','B'],'risk_score':[1,0],'predicted_behavior':['soft_play','none']})
  for k in range(1,6):truth[f'evidence_hand_{k}']=[f'H{k}','NO_EVIDENCE']
  return truth,truth.copy()
 def test_evidence_credit_independent_of_risk(self):
  truth,pred=self.fixture();pred['risk_score']=[0.,1.]
  full=score(truth,pred,'pair_id')
  for k in range(1,6):pred[f'evidence_hand_{k}']='NO_EVIDENCE'
  self.assertAlmostEqual(full-score(truth,pred,'pair_id'),.2)
 def test_duplicate_evidence_rejected(self):
  truth,pred=self.fixture();pred.loc[0,'evidence_hand_2']='H1'
  with self.assertRaises(ParticipantVisibleError):score(truth,pred,'pair_id')

class ContextInvariant(unittest.TestCase):
 def fixture(self):
  ss=[];aa=[]
  for hid in [1,2]:
   winner='B' if hid==1 else 'A'
   for p,card in [('A','2h'),('B','As')]:
    win=p==winner
    ss.append(dict(hid=hid,player_id=p,hole_card_1=card,hole_card_2='7d' if p=='A' else 'Ks',net_chips=2 if win else -2,folded=not win,won_share=float(win)))
   loser='A' if winner=='B' else 'B'
   for n,(p,act,amount,call,pot) in enumerate([(loser,'call',2,2,3),(winner,'raise',6,2,5),(loser,'fold',0,4,11)]):
    aa.append(dict(hid=hid,player_id=p,street='preflop',action_no=n,action=act,amount=amount,to_call=call,pot_before=pot))
  hands=pl.DataFrame({'hid':[1,2],'hand_id':['H1','H2'],'big_blind':[2,2],'board_cards':['','']})
  return pl.DataFrame(ss),pl.DataFrame(aa),hands
 def compute(self,s,a,h,p='A',q='B'):
  with patch('context_features.pl.read_parquet',side_effect=lambda path:s if 'seats' in path else a):
   return extract(0,pl.DataFrame({'player_1':[p],'player_2':[q],'pair_id':['PAIR']}),h).sort('hand_id')
 def test_pair_order_and_id_renaming(self):
  s,a,h=self.fixture();base=self.compute(s,a,h)
  self.assertTrue(base.equals(self.compute(s,a,h,'B','A')))
  rename={'A':'Z','B':'Y'}
  s=s.with_columns(pl.col('player_id').replace_strict(rename));a=a.with_columns(pl.col('player_id').replace_strict(rename))
  np.testing.assert_allclose(base.select(pl.exclude('pair_id','hand_id')).to_numpy(),self.compute(s,a,h,'Z','Y').select(pl.exclude('pair_id','hand_id')).to_numpy())
 def test_immediate_yield_detected(self):
  s,a,h=self.fixture();r=self.compute(s,a,h)
  self.assertEqual(r['cx_yield_immediate'].to_list(),[1.,1.])
  self.assertEqual(r['cx_pre_partner_fold'].to_list(),[1.,1.])
if __name__=='__main__':unittest.main()
