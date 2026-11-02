import numpy as np, sys
sys.path.insert(0,'.')
from figs_common import *
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from sadrl.envs import DynamicMaze
# --- maze figure
e=DynamicMaze(); e.reset()
fig,ax=plt.subplots(figsize=(COL1*0.62,COL1*0.62))
ax.grid(False)
for r in range(e.H):
    for c in range(e.W):
        ax.add_patch(Rectangle((c,e.H-1-r),1,1,fc='#2b2b2a' if e.walls[r,c] else '#fcfcfb',ec='#e4e3df',lw=0.3))
for j,tr in enumerate(e.tracks):
    cs=[p[1] for p in tr]; rs=[p[0] for p in tr]
    ax.plot([cs[0]+0.5]*2,[e.H-1-min(rs)+0.5,e.H-1-max(rs)+0.5],color='#e34948',lw=0.8,ls=':')
    p=tr[e.idx[j]]; ax.add_patch(Circle((p[1]+0.5,e.H-1-p[0]+0.5),0.33,fc='#e34948',ec='none'))
    ax.annotate('',xy=(p[1]+0.5,e.H-1-p[0]+1.05),xytext=(p[1]+0.5,e.H-1-p[0]+0.6),arrowprops=dict(arrowstyle='-|>',color='#e34948',lw=0.8))
s,g=e.start,e.goal
ax.text(s[1]+0.5,e.H-1-s[0]+0.5,'S',ha='center',va='center',fontsize=8,weight='bold',color='#2a78d6')
ax.text(g[1]+0.5,e.H-1-g[0]+0.5,'G',ha='center',va='center',fontsize=8,weight='bold',color='#008300')
ax.set_xlim(0,e.W);ax.set_ylim(0,e.H);ax.set_aspect('equal');ax.axis('off')
fig.savefig('figs/maze.pdf'); plt.close(fig)

# --- acceptance probability illustration
fig,axs=plt.subplots(1,2,figsize=(COL1,1.9))
d=np.linspace(0,4,200)
for T,col in zip([4,1,0.25,0.05],['#c9dcf5','#7fb0ea','#2a78d6','#0d3f80']):
    axs[0].plot(d,np.exp(-d/T),color=col,label=f'$T={T}$')
axs[0].set_xlabel(r'normalised gap $\tilde\Delta$'); axs[0].set_ylabel('acceptance prob.')
axs[0].legend(loc='upper right',handlelength=1.0,fontsize=6,labelspacing=0.2)
axs[0].set_title('(a) Metropolis acceptance',loc='left')
# example Q values, induced action distribution
q=np.array([1.0,0.9,0.2,-0.6]); sig=np.mean(q.max()-q[1:])
w=0.2; x=np.arange(4)
def mae_probs(T):
    a=np.argmax(q); p=np.zeros(4)
    for j in range(4):
        if j==a: continue
        p[j]=(1/3)*np.exp(-(q[a]-q[j])/(sig*T))
    p[a]=1-p.sum(); return p
eps=0.6
pe=np.full(4,eps/4); pe[0]+=1-eps
from scipy.optimize import brentq
Tm=brentq(lambda T:(1-mae_probs(T)[0])-0.45,0.01,100)
pm=mae_probs(Tm)
axs[1].bar(x-w/2-0.02,pe,w,color=C['eps'],label=r'$\epsilon$-greedy')
axs[1].bar(x+w/2+0.02,pm,w,color=C['mae'],label='MAE')
axs[1].set_xticks(x,[f'$a_{i+1}$\n({q[i]:+.1f})' for i in range(4)]); axs[1].set_xlabel('action (Q-value)'); axs[1].set_ylabel('selection prob.')
axs[1].legend(fontsize=6,loc='upper right'); axs[1].set_title('(b) equal non-greedy rate',loc='left')
fig.tight_layout(w_pad=1.0); fig.savefig('figs/acceptance.pdf'); plt.close(fig)
print(pe.round(3),pm.round(3), 1-pe[0], 1-pm[0])
