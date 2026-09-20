"""Build the README figure from committed report JSON. No competition data needed."""
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK,ACC,MUT='#1b2430','#c2410c','#94a3b8'
# name, public score, kind: 0=baseline 1=three changes at once 2=single variable
sub=[('v1',.83232,0),('v2',.83492,1),('v3',.83472,1),('v4',.83942,2),
     ('v5',.84409,2),('v6',.84436,2),('v7',.84537,2)]
cen=json.load(open('reports/risk_blend_review.json'))['v2_reference']['score_dependent_censor_sensitivity']

fig,(a,b)=plt.subplots(1,2,figsize=(11,4.1))

x=range(len(sub))
a.plot(x,[s[1] for s in sub],'-',color=MUT,lw=1.5,zorder=1)
COL={0:INK,1:MUT,2:ACC}
for i,(n,v,kind) in enumerate(sub):
    a.scatter(i,v,s=70,zorder=3,color=COL[kind],edgecolor='white',lw=1.2)
a.axhline(.8034,ls='--',lw=1,color=INK,alpha=.45)
a.text(6.1,.8043,'field median  0.8034',fontsize=8,color=INK,alpha=.7,ha='right')
a.set_xticks(list(x));a.set_xticklabels([s[0] for s in sub])
a.set_ylabel('public leaderboard score')
a.set_title('One variable per submission',loc='left',fontsize=11,color=INK)
a.scatter([],[],s=70,color=INK,label='first end-to-end pipeline')
a.scatter([],[],s=70,color=MUT,label='three changes at once  (0/2 improved)')
a.scatter([],[],s=70,color=ACC,label='exactly one change  (4/4 improved)')
a.legend(frameon=False,fontsize=8,loc='upper left')
a.set_ylim(.8008,.8495)

ks=[0,50,100,200,400];ys=[cen[f'censor_{k}'] for k in ks]
b.plot(ks,ys,'-o',color=ACC,lw=2,ms=6,zorder=3)
b.axhline(.905,ls='--',lw=1,color=INK,alpha=.45)
b.text(196,.9105,'implied by leaderboard  ~0.90',fontsize=8,color=INK,alpha=.7)
b.annotate('naive local score\nsays 0.72',xy=(0,ys[0]),xytext=(48,.745),fontsize=8,
           color=INK,arrowprops=dict(arrowstyle='->',color=INK,alpha=.5,lw=.9))
b.set_xlabel('unlabelled pairs excluded from scoring')
b.set_ylabel('average precision')
b.set_title('The local metric was measuring the wrong problem',loc='left',fontsize=11,color=INK)
b.set_ylim(.68,1.0)

for ax in (a,b):
    ax.grid(axis='y',alpha=.18,lw=.7)
    for s in ('top','right'):ax.spines[s].set_visible(False)
    for s in ('left','bottom'):ax.spines[s].set_color(MUT)
    ax.tick_params(colors=INK,labelsize=8.5)

fig.tight_layout()
fig.savefig('assets/results.png',dpi=200,facecolor='white')
print('wrote assets/results.png')
