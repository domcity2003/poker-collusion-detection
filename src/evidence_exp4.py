"""Graded relevance: the organisers' own evidence_rank is a severity ordering.

fold_better_sum falls monotonically from rank 1 to rank 5, so rank 1 hands are
the most detectable, not merely the first listed. Training YetiRank on a binary
target throws that away. MAP@5 rewards putting true evidence early, so grading
the target by 6-rank should align the ranker with the metric.
"""
import sys,json,pickle
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostRanker,Pool
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evidence_exp import map5,znorm,ID,FAM

def run(h,fc,graded,tag,probs,loss='YetiRank'):
 X=np.nan_to_num(h.select(fc).to_numpy().astype('float32'),nan=0,posinf=1e6,neginf=-1e6)
 y=h['ev'].to_numpy()
 rel=(6-h['evidence_rank'].fill_null(6).to_numpy()).astype(float) if graded else y.astype(float)
 rel=np.where(y==1,rel,0.)
 fold=h['fold'].to_numpy();pair=h['pair_id'].to_numpy();fam=h['behavior_family'].to_numpy()
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
   m=CatBoostRanker(loss_function=loss,iterations=600,depth=5,learning_rate=.06,
                    thread_count=6,verbose=False,random_seed=2026,allow_writing_files=False)
   m.fit(Pool(XX[i],rel[i],group_id=gid[i]));spec.append(m.predict(XX[te]))
  spec=np.vstack(spec);tep=pair[te]
  for p in np.unique(tep):
   idx=np.flatnonzero(tep==p);w=probs.get(p,np.ones(3)/3);b=spec[:,idx]
   r=np.vstack([(np.argsort(np.argsort(x))+1)/len(x) for x in b])
   scores[te[idx]]=(w[:,None]*r).sum(0)
 d=h.with_columns(pl.Series('s',scores)).filter(pl.col('fold').is_in([1,2,3,4]))
 m5,_=map5(d,'s')
 out={'variant':tag,'map5':m5,'per_family':{fm:map5(d.filter(pl.col('behavior_family')==fm),'s')[0] for fm in FAM}}
 print(json.dumps(out),flush=True);return out

def main():
 h=pl.read_parquet('artifacts/dev_hands_hindsight.parquet')
 ev=pl.read_csv('data/development_evidence.csv')
 h=h.join(ev.select('pair_id','hand_id','evidence_rank'),on=['pair_id','hand_id'],how='left').sort('pair_id')
 probs=pickle.load(open('artifacts/fam_probs.pkl','rb'))
 fc=[c for c,t in h.schema.items() if t.is_numeric() and c not in ID]
 res=[run(h,fc,False,'binary',probs),run(h,fc,True,'graded',probs),
      run(h,fc,True,'graded+ndcg',probs,'NDCG')]
 Path('reports/evidence_graded.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
