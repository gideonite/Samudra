import json,glob,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/home/user/Samudra/experiments/pcgb_toy/runs/'; O='/home/user/Samudra/experiments/pcgb_toy/figs/'
C={'baseline':'#1f77b4','stochdepth':'#2ca02c','pcgb':'#d62728'}
def rows(t): return [json.loads(l) for l in open(R+t+'/log.jsonl')]
# Fig A: mask oscillation
fig,ax=plt.subplots(1,2,figsize=(11,3.4))
r=rows('calib_ks1d_pcgb'); rd=[x for x in r if x['type']=='round']; ev=[x for x in r if x['type']=='eval']
kept=[bin(x['mask']).count('1') for x in rd]; t=[x['elapsed'] for x in rd]
a=ax[0]; a.step(t,kept,where='post',color='#888',lw=1.5); a.set_ylabel('skips kept in round (0-4)'); a.set_xlabel('wall-clock (s)')
b=a.twinx(); b.plot([e['elapsed'] for e in ev],[e['metric'] for e in ev],'o-',color=C['pcgb']); b.set_ylabel('val 10-step nRMSE (unmasked net)',color=C['pcgb']); b.set_yscale('log')
a.set_title('(a) 12-min v1 run: KS1D n=512, seed 0',fontsize=10)
rr=[x for x in rows('ks1d_pcgb_n512_s0_c43f057_df5512') if x['type']=='round'][:80]
a=ax[1]; a.step(range(len(rr)),[bin(x['mask']).count('1') for x in rr],where='post',color='#888',lw=1.2); a.set_xlabel('round (first 80 of 300)'); a.set_ylabel('skips kept in round (0-4)')
a.set_title('(b) 120-min run (300 rounds): same 0 <-> 4 alternation',fontsize=10)
plt.tight_layout(); plt.savefig(O+'mask_oscillation_ks1d_n512.png',dpi=150); plt.close()
# Fig B: 10x curves
fig,ax=plt.subplots(1,2,figsize=(11,3.6))
for a,ds,title in [(ax[0],'ks1d','KS1D n=512'),(ax[1],'kolmo2d','Kolmo2D n=512')]:
    for arm in ['baseline','stochdepth','pcgb']:
        cs=[]
        for s in range(3):
            f=glob.glob(R+f'{ds}_{arm}_n512_s{s}_*_{"df5512" if arm=="pcgb" else "2ae778"}')
            ev=[x for x in rows(f[0].split('/')[-1]) if x['type']=='eval']; cs.append(([e['elapsed']/60 for e in ev],[e['metric'] for e in ev]))
        n=min(len(c[0]) for c in cs); x=np.mean([c[0][:n] for c in cs],0); Y=np.array([c[1][:n] for c in cs])
        a.plot(x,Y.mean(0),'-o',ms=3,color=C[arm],label=arm); a.fill_between(x,Y.min(0),Y.max(0),color=C[arm],alpha=.18)
    a.set_yscale('log'); a.set_xlabel('wall-clock (min)'); a.set_ylabel('val 10-step nRMSE'); a.set_title(title,fontsize=10); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(O+'budget10x_curves_n512.png',dpi=150); plt.close()
