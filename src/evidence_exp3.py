"""Does the hindsight family lift evidence retrieval? Same protocol as exp2."""
import sys,json
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostClassifier,CatBoostRanker,Pool
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evidence_exp import map5,znorm,ID,FAM

def run(h,fc,tag,probs):
 X=np.nan_to_num(h.select(fc).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 y=h['ev'].to_numpy();fold=h['fold'].to_numpy();pair=h['pair_id'].to_numpy();fam=h['behavior_family'].to_numpy()
 g={}
 for i,p in enumerate(pair):g.setdefault(p,[]).append(i)
 g={k:np.array(v) for k,v in g.items()}
 XX=np.hstack([X,znorm(X,g)])
 gid=np.searchsorted(np.unique(pair),pair)
 scores=np.zeros(len(y))
 for f in [1,2,3,4]:
  tr=(fold!=f)&(fold!=0);te=np.flatnonzero(fold==f)
  spec=[]
  for fm in FAM:
   s=tr&(fam==fm);o=np.argsort(gid[s],kind='stable');i=np.flatnonzero(s)[o]
   m=CatBoostRanker(loss_function='YetiRank',iterations=600,depth=5,learning_rate=.06,thread_count=6,verbose=False,random_seed=2026,allow_writing_files=False)
   m.fit(Pool(XX[i],y[i],group_id=gid[i]));spec.append(m.predict(XX[te]))
  spec=np.vstack(spec);tep=pair[te]
  for p in np.unique(tep):
   idx=np.flatnonzero(tep==p);w=probs.get(p,np.ones(3)/3);b=spec[:,idx]
   r=np.vstack([(np.argsort(np.argsort(x))+1)/len(x) for x in b])
   scores[te[idx]]=(w[:,None]*r).sum(0)
 d=h.with_columns(pl.Series('s',scores)).filter(pl.col('fold').is_in([1,2,3,4]))
 m5,n=map5(d,'s')
 out={'variant':tag,'map5':m5,'n_features':len(fc),
      'per_family':{fm:map5(d.filter(pl.col('behavior_family')==fm),'s')[0] for fm in FAM}}
 print(json.dumps(out),flush=True);return out

def main():
 h=pl.read_parquet('artifacts/dev_hands_hindsight.parquet').sort('pair_id')
 hs=[c for c,t in h.schema.items() if t.is_numeric() and c not in ID]
 import pickle
 probs=pickle.load(open('artifacts/fam_probs.pkl','rb'))
 base=[c for c in hs if not c.startswith(('p_would','q_would','pair_','best_','winner_','partner_paid','acts_with','fold_with','fold_beating','passive_','aggressive_'))]
 res=[run(h,base,'base(no hindsight)',probs),run(h,hs,'with hindsight',probs)]
 Path('reports/evidence_hindsight.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
