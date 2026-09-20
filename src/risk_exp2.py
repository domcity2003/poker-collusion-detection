import sys,json,argparse
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0,str(Path(__file__).resolve().parent))
from risk_exp import load,cv,ID
from relative import add_relative,signal_columns
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from official_metric import _average_precision

def poolblend(df,oof,a=.15):
 t=df.select('pool').with_columns(pl.Series('risk',oof))
 t=t.with_columns(((pl.col('risk')-pl.col('risk').mean())/(pl.col('risk').std()+1e-9)).over('pool').alias('z'))
 z=t['z'].to_numpy();zz=(z-z.min())/(z.max()-z.min()+1e-9)
 return (1-a)*oof+a*zz

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--k',type=int,default=60);args=ap.parse_args()
 df=load()
 cols=signal_columns(df,args.k)
 df=add_relative(df,cols)
 print('with relative',df.shape,flush=True)
 allcols=[c for c,t in df.schema.items() if c not in ID and t.is_numeric()]
 r,oof=cv(df,allcols,tag='rich+relative');print(json.dumps(r),flush=True)
 lab=df['label'].to_numpy();fold=df['fold'].to_numpy();sel=np.isin(fold,[1,2,3,4]);y=(lab==1).astype(int)
 bl=poolblend(df,oof)
 r['broad_ap_poolblend']=_average_precision(y[sel],bl[sel])
 print('poolblend',r['broad_ap_poolblend'],flush=True)
 df.select('player_1','player_2','pool','fold','label').with_columns(pl.Series('risk',oof),pl.Series('risk_blend',bl)).write_parquet('artifacts/risk_oof_relative.parquet')
 Path('reports/risk_relative.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
