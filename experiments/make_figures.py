import pickle, numpy as np, os, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
BLUE,ORANGE,AQUA,INK,MUTE,GRID='#2a78d6','#eb6834','#1baf7a','#0b0b0b','#52514e','#e4e3df'
SEQ=LinearSegmentedColormap.from_list('seq',['#ffffff','#cde2fb','#86b6ef','#2a78d6','#0d366b'])
DIV=LinearSegmentedColormap.from_list('div',['#b2431a','#eb6834','#f0efec','#2a78d6','#104281'])
ORD=['#cde2fb','#5598e7','#1c5cab','#0d366b']
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':MUTE,'axes.labelcolor':INK,'xtick.color':MUTE,'ytick.color':MUTE,'axes.titlesize':13,'axes.titleweight':'bold','xtick.labelsize':11,'ytick.labelsize':11,'axes.labelsize':12,'legend.fontsize':11,'axes.titlelocation':'left','pdf.fonttype':42,'font.family':'DejaVu Sans','legend.frameon':False})
PROT="ACDEFGHIKLMNPQRSTVWY"; OUT='/mnt/user-data/outputs/paper/'
def save(fig,name): fig.savefig(OUT+name+'.pdf',bbox_inches='tight',dpi=240); fig.savefig(OUT+name+'.png',dpi=200,bbox_inches='tight'); plt.close(fig); print('saved',name)
G={s:pickle.load(open(f'gb1_{s}.pkl','rb')) for s in ['one_vs_rest','two_vs_rest','three_vs_rest','low_vs_high','sampled'] if os.path.exists(f'gb1_{s}.pkl')}
NICE={'one_vs_rest':'one-vs-rest','two_vs_rest':'two-vs-rest','three_vs_rest':'three-vs-rest','low_vs_high':'low-vs-high','sampled':'sampled'}
SITES=['V39','D40','G41','V54']
# ---------- Fig GB1
if G:
    main='sampled' if 'sampled' in G else 'three_vs_rest'; R=G[main]['res']
    fig,ax=plt.subplots(2,3,figsize=(13.5,8.4)); ax=ax.ravel()
    # (a) order shares
    a=ax[0]; names=list(G); x=np.arange(len(names)); w=0.36
    for k,(key,off) in enumerate([('order_linear',-w/2-0.02),('order_gauss',w/2+0.02)]):
        bot=np.zeros(len(names))
        for r in range(4):
            v=np.array([G[s]['res'][key][r] for s in names]); a.bar(x+off,v,w,bottom=bot,color=ORD[r],edgecolor='white',linewidth=1.2,label=f'order {r+1}' if k==0 else None); bot+=v
    for i in x: a.text(i-w/2-0.02,1.02,'L',ha='center',fontsize=10,color=MUTE); a.text(i+w/2+0.02,1.02,'G',ha='center',fontsize=10,color=MUTE)
    a.set_xticks(x); a.set_xticklabels([NICE[s] for s in names],rotation=20,ha='right'); a.set_ylabel('share of landscape variance'); a.set_ylim(0,1.12); a.legend(ncol=4,loc='upper center',bbox_to_anchor=(0.5,-0.28),fontsize=10.5,handlelength=1); a.set_title('a')
    # (b) pairwise share matrix
    a=ax[1]; sh=R['anova_gauss']; P=np.full((4,4),np.nan)
    for i in range(4):
        P[i,i]=sh[str((i,))]
        for j in range(i+1,4): P[i,j]=P[j,i]=sh[str((i,j))]
    im=a.imshow(P,cmap=SEQ,vmin=0); a.set_xticks(range(4)); a.set_yticks(range(4)); a.set_xticklabels(SITES); a.set_yticklabels(SITES)
    for i in range(4):
        for j in range(4): a.text(j,i,f'{100*P[i,j]:.1f}',ha='center',va='center',color='white' if P[i,j]>0.6*np.nanmax(P) else INK,fontsize=11)
    a.spines[:].set_visible(False); a.set_title('b')
    # (c) held-out epistasis
    a=ax[2]; D=np.array([r for r in R['doubles'] if r[5]],dtype=float)
    if len(D):
        a.scatter(D[:,2],D[:,4],s=9,color=ORANGE,alpha=.55,linewidths=0,label='linear W-PSSKM'); a.scatter(D[:,2],D[:,3],s=9,color=BLUE,alpha=.6,linewidths=0,label='Gaussian W-PSSKM')
        lim=[min(D[:,2].min(),D[:,3].min()),max(D[:,2].max(),D[:,3].max())]; a.plot(lim,lim,color=MUTE,lw=.8,ls='--'); e=R.get('epi_heldout',{})
        a.set_xlabel('measured epistasis  $y_{ab}-y_a-y_b+y_{wt}$'); a.set_ylabel('predicted epistasis'); a.legend(fontsize=10.5,loc='upper left',markerscale=1.6)
        a.set_title('c')
    # (d) strongest pair interaction map
    a=ax[3]; pairs=[(i,j) for i in range(4) for j in range(i+1,4)]; bp=max(pairs,key=lambda p:sh[str(p)]); T=G[main]['main'][bp]; v=np.abs(T).max()
    im=a.imshow(T,cmap=DIV,norm=TwoSlopeNorm(0,-v,v),aspect='auto'); a.set_xticks(range(20)); a.set_xticklabels(list(PROT),fontsize=10); a.set_yticks(range(20)); a.set_yticklabels(list(PROT),fontsize=10)
    a.set_xlabel(f'residue at {SITES[bp[1]]}'); a.set_ylabel(f'residue at {SITES[bp[0]]}'); a.spines[:].set_visible(False); plt.colorbar(im,ax=a,fraction=0.046,pad=0.03); a.set_title('d')
    # (e) dial
    a=ax[4]; cols=[MUTE,ORANGE,BLUE,AQUA,'#6250d6']
    for c,s in zip(cols,names):
        dl=G[s]['res']['dial']; gs=sorted(dl,key=float); a.plot([float(g) for g in gs],[dl[g]['test'] for g in gs],color=c,lw=1.6,marker='o',ms=3); a.scatter([float(gs[0])/2.2],[G[s]['res']['linear']['test']],color=c,marker='s',s=18,zorder=3)
        a.text(float(gs[-1])*1.12,dl[gs[-1]]['test'],NICE[s],color=INK,fontsize=10,va='center')
    a.set_xscale('log'); a.set_xlim(0.01,40); a.set_xlabel('bandwidth $g$ (squares at left: linear kernel)'); a.set_ylabel(r'test Spearman $\rho$'); a.grid(axis='y',color=GRID,lw=.6); a.set_title('e')
    # (f) main effects vs measured singles
    a=ax[5]; S=np.array([r[:5] for r in R['singles']],dtype=float)
    a.scatter(S[:,2],S[:,3],s=14,color=BLUE,alpha=.75,linewidths=0); lim=[0,max(S[:,2].max(),S[:,3].max())]; a.plot(lim,lim,color=MUTE,lw=.8,ls='--')
    from scipy.stats import spearmanr
    a.set_xlabel('measured single-mutant fitness'); a.set_ylabel('model landscape value'); a.set_title('f')
    fig.tight_layout(pad=1.2,w_pad=2.0,h_pad=2.2); save(fig,'fig_interp_gb1')
# ---------- Fig DNA promoters
p=pickle.load(open('pos_gb_nontata.pkl','rb')); imp=np.array(p['res']['imp']); M=np.array(p['res']['M'])[:,:,1]; M=M-M.mean(1,keepdims=True)
fig,ax=plt.subplots(1,2,figsize=(13.5,3.8),gridspec_kw={'width_ratios':[1.5,1]})
pos=np.arange(251)-200; pos=np.where(pos>=0,pos+1,pos); a=ax[0]; a.bar(np.arange(251)-200,imp/imp.max(),width=1.0,color=BLUE); a.set_xlabel('position relative to annotated TSS'); a.set_ylabel('knock-out importance (rel.)'); a.set_xticks([-200,-150,-100,-50,0,50]); a.set_xticklabels(['-200','-150','-100','-50','TSS','+50']); a.set_title('a')
a=ax[1]; W=M[190:211].T; v=np.abs(W).max(); im=a.imshow(W,cmap=DIV,norm=TwoSlopeNorm(0,-v,v),aspect='auto'); a.set_yticks(range(4)); a.set_yticklabels(list('ACGT')); a.set_xticks(range(0,21,2)); a.set_xticklabels([str(pos[190+i]) if pos[190+i]<0 else f'+{pos[190+i]}' for i in range(0,21,2)],fontsize=10); a.spines[:].set_visible(False); a.set_xlabel('position'); plt.colorbar(im,ax=a,fraction=0.046,pad=0.03,label='attribution to promoter class'); a.set_title('b')
fig.tight_layout(pad=1.2,w_pad=2.0,h_pad=2.2); save(fig,'fig_interp_promoter')
# ---------- Fig DeepLoc
LOCS=['Cytoplasm','Nucleus','Extracellular','Cell membrane','Mitochondrion','Plastid','Endoplasmic reticulum','Lysosome/Vacuole','Golgi apparatus','Peroxisome']
p=pickle.load(open('pos_deeploc.pkl','rb')); imp=np.array(p['res']['imp']); M=np.array(p['res']['M'])
fig,ax=plt.subplots(1,2,figsize=(13.5,4.1),gridspec_kw={'width_ratios':[1,1.5]})
a=ax[0]; a.bar(np.arange(1,121),imp[:120]/imp.max(),width=1.0,color=BLUE); a.set_xlabel('residue position from N-terminus'); a.set_ylabel('knock-out importance (rel.)'); a.set_title('a')
sel=[2,4,5,1,3,6]; W=np.stack([M[1:25,:,c].sum(0) for c in sel]); W=W-W.mean(1,keepdims=True); order=np.argsort(-W[0]); v=np.abs(W).max()
a=ax[1]; im=a.imshow(W[:,order],cmap=DIV,norm=TwoSlopeNorm(0,-v,v),aspect='auto'); a.set_yticks(range(len(sel))); a.set_yticklabels([LOCS[c] for c in sel]); a.set_xticks(range(20)); a.set_xticklabels([PROT[i] for i in order]); a.spines[:].set_visible(False); plt.colorbar(im,ax=a,fraction=0.03,pad=0.02,label='attribution'); a.set_title('b')
fig.tight_layout(pad=1.2,w_pad=2.0,h_pad=2.2); save(fig,'fig_interp_deeploc')
json.dump({'deeploc_W':{LOCS[c]:{PROT[i]:float(W[k,i]) for i in range(20)} for k,c in enumerate(sel)}},open('deeploc_attr.json','w'))
# ---------- Fig overview: profiles + dials for all datasets
DS=[('gb_nontata','Promoters (non-TATA)'),('gb_cohn','Enhancers (Cohn)'),('deeploc','DeepLoc'),('meltome_mixed_split','Meltome'),('tape_rh_family','Remote homology (family)')]
DS=[d for d in DS if os.path.exists(f'pos_{d[0]}.pkl')]
fig,ax=plt.subplots(2,len(DS),figsize=(13.5,5.4))
for k,(n,lab) in enumerate(DS):
    r=pickle.load(open(f'pos_{n}.pkl','rb'))['res']; imp=np.array(r['imp']); a=ax[0,k]; a.bar(np.arange(len(imp))*r['window'],imp/imp.max(),width=r['window'],color=BLUE,rasterized=True); a.set_title(lab,fontsize=11); a.set_xlabel('position',fontsize=10.5); 
    if k==0: a.set_ylabel('knock-out importance')
    a=ax[1,k]; gs=sorted(r['dial_test'],key=float); a.plot([float(g) for g in gs],[r['dial_test'][g] for g in gs],color=BLUE,lw=1.6,marker='o',ms=3); a.axhline(r['linear_test'],color=ORANGE,lw=1.2,ls='--'); a.set_xscale('log'); a.set_xlabel('bandwidth $g$',fontsize=10.5); a.grid(axis='y',color=GRID,lw=.6)
    if k==0: a.set_ylabel('test score'); 
fig.tight_layout(pad=1.2,w_pad=2.0,h_pad=2.2); save(fig,'fig_interp_all')
