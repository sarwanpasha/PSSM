"""Regenerate the GB1 interpretability figure for the LM-weighted kernel (gb1lm_*.pkl)."""
import pickle, numpy as np, itertools
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import spearmanr
BLUE,ORANGE,AQUA,INK,MUTE,GRID='#2a78d6','#eb6834','#1baf7a','#0b0b0b','#52514e','#e4e3df'
SEQ=LinearSegmentedColormap.from_list('seq',['#ffffff','#cde2fb','#86b6ef','#2a78d6','#0d366b'])
DIV=LinearSegmentedColormap.from_list('div',['#b2431a','#eb6834','#f0efec','#2a78d6','#104281'])
ORD=['#cde2fb','#5598e7','#1c5cab','#0d366b']
plt.rcParams.update({'pdf.compression':9,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':MUTE,'axes.labelcolor':INK,'xtick.color':MUTE,'ytick.color':MUTE,'axes.titlesize':13,'axes.titleweight':'bold','xtick.labelsize':11,'ytick.labelsize':11,'axes.labelsize':12,'legend.fontsize':11,'axes.titlelocation':'left','pdf.fonttype':42,'font.family':'DejaVu Sans','legend.frameon':False})
PROT="ACDEFGHIKLMNPQRSTVWY"; OUT='/mnt/user-data/outputs/paper/'; SITES=['V39','D40','G41','V54']
NICE={'two_vs_rest':'two-vs-rest','three_vs_rest':'three-vs-rest','sampled':'sampled'}
G={s:pickle.load(open(f'gb1lm_{s}.pkl','rb')) for s in NICE}
OLD={s:pickle.load(open(f'gb1_{s}.pkl','rb')) for s in NICE}      # for the linear-kernel reference
main='sampled'; R=G[main]['res']; names=['two_vs_rest','three_vs_rest','sampled']
fig,ax=plt.subplots(2,3,figsize=(13.5,8.4)); ax=ax.ravel()
a=ax[0]; x=np.arange(len(names)); w=0.36
for k,(key,off,src) in enumerate([('order_linear',-w/2-0.02,OLD),('order',w/2+0.02,G)]):
    bot=np.zeros(len(names))
    for r in range(4):
        v=np.array([src[s]['res'][key][r] for s in names])
        a.bar(x+off,v,w,bottom=bot,color=ORD[r],edgecolor='white',linewidth=1.2,label=f'order {r+1}' if k==0 else None); bot+=v
for i in x: a.text(i-w/2-0.02,1.02,'L',ha='center',fontsize=10,color=MUTE); a.text(i+w/2+0.02,1.02,'G',ha='center',fontsize=10,color=MUTE)
a.set_xticks(x); a.set_xticklabels([NICE[s] for s in names],rotation=20,ha='right'); a.set_ylabel('share of landscape variance')
a.set_ylim(0,1.12); a.legend(ncol=4,loc='upper center',bbox_to_anchor=(0.5,-0.3),fontsize=10.5,handlelength=1); a.set_title('a')
a=ax[1]; sh=R['anova']; P=np.full((4,4),np.nan)
for i in range(4):
    P[i,i]=sh[str((i,))]
    for j in range(i+1,4): P[i,j]=P[j,i]=sh[str((i,j))]
im=a.imshow(P,cmap=SEQ,vmin=0,rasterized=True); a.set_xticks(range(4)); a.set_yticks(range(4)); a.set_xticklabels(SITES); a.set_yticklabels(SITES)
for i in range(4):
    for j in range(4): a.text(j,i,f'{100*P[i,j]:.1f}',ha='center',va='center',color='white' if P[i,j]>0.6*np.nanmax(P) else INK,fontsize=11)
a.spines[:].set_visible(False); a.set_title('b')
a=ax[2]; D=np.array([r[:4] for r in R['doubles'] if r[4]],dtype=float); Dl=np.array([r for r in OLD[main]['res']['doubles'] if r[5]],dtype=float)
a.scatter(Dl[:,2],Dl[:,4],s=9,color=ORANGE,alpha=.55,linewidths=0,label='linear kernel')
a.scatter(D[:,2],D[:,3],s=9,color=BLUE,alpha=.7,linewidths=0,label='Gaussian, LM weights')
lim=[min(D[:,2].min(),D[:,3].min()),max(D[:,2].max(),D[:,3].max())]; a.plot(lim,lim,color=MUTE,lw=.8,ls='--')
a.set_xlabel('measured epistasis  $y_{ab}-y_a-y_b+y_{wt}$'); a.set_ylabel('predicted epistasis'); a.legend(fontsize=10.5,loc='upper left',markerscale=1.6)
a.set_title('c')
a=ax[3]; pairs=[(i,j) for i in range(4) for j in range(i+1,4)]; bp=max(pairs,key=lambda p:sh[str(p)]); T=G[main]['main'][bp]; v=np.abs(T).max()
im=a.imshow(T,cmap=DIV,norm=TwoSlopeNorm(0,-v,v),aspect='auto',rasterized=True); a.set_xticks(range(20)); a.set_xticklabels(list(PROT),fontsize=10); a.set_yticks(range(20)); a.set_yticklabels(list(PROT),fontsize=10)
a.set_xlabel(f'residue at {SITES[bp[1]]}'); a.set_ylabel(f'residue at {SITES[bp[0]]}'); a.spines[:].set_visible(False); plt.colorbar(im,ax=a,fraction=0.046,pad=0.03)
a.set_title('d')
a=ax[4]; cols=[ORANGE,BLUE,AQUA]
for c,s in zip(cols,names):
    dl=G[s]['res']['dial']; gs=sorted(dl,key=float)
    a.plot([float(g) for g in gs],[dl[g][1] for g in gs],color=c,lw=1.6,marker='o',ms=3)
    a.text(float(gs[-1])*1.1,dl[gs[-1]][1],NICE[s],color=INK,fontsize=10,va='center')
a.set_xscale('log'); a.set_xlabel('bandwidth $g$'); a.set_ylabel(r'test Spearman $\rho$'); a.grid(axis='y',color=GRID,lw=.6); a.set_title('e')
a=ax[5]; S=np.array([r[2:4] for r in R['singles']],dtype=float)
a.scatter(S[:,0],S[:,1],s=14,color=BLUE,alpha=.75,linewidths=0); lim=[0,max(S[:,0].max(),S[:,1].max())]; a.plot(lim,lim,color=MUTE,lw=.8,ls='--')
a.set_xlabel('measured single-mutant fitness'); a.set_ylabel('model landscape value')
a.set_title('f')
fig.tight_layout(pad=1.2,w_pad=2.0,h_pad=2.2); fig.savefig(OUT+'fig_interp_gb1.pdf',bbox_inches='tight',dpi=150); fig.savefig(OUT+'fig_interp_gb1.png',dpi=200,bbox_inches='tight')
print('saved; orders',[round(x,3) for x in R['order']],'best pair',SITES[bp[0]],SITES[bp[1]],round(sh[str(bp)],3))
