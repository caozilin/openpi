from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import weighted_window

root=Path(__file__).resolve().parent
curves=json.loads((root/'parsed_curves.json').read_text())
summary=json.loads((root/'comparison.json').read_text())
small_end=curves['bs24']['metrics'][-1]['samples']
fig,axes=plt.subplots(1,3,figsize=(14,4.2))
colors={'bs84':'#2864ac','bs24':'#d27521'}
labels={'bs84':'7 GPUs, batch 84','bs24':'6 GPUs, batch 24'}
for name,run in curves.items():
 points=[p for p in run['metrics'] if p['samples']<=small_end*1.25]
 x=np.array([p['samples'] for p in points]);y=np.array([p['metrics']['loss/action'] for p in points])
 axes[0].plot(x,y,alpha=.35,color=colors[name],lw=1)
 if len(y)>=3:
  ends=np.arange(1680,x[-1]+1,120)
  smooth=[weighted_window(run,float(end-1680),float(end)) for end in ends]
  axes[0].plot(ends,smooth,label=labels[name],color=colors[name],lw=2)
 points=[p for p in run['metrics'] if p['step']<=curves['bs24']['metrics'][-1]['step']]
 x=np.array([p['step'] for p in points]);y=np.array([p['metrics']['loss/action'] for p in points])
 axes[1].plot(x,y,alpha=.3,color=colors[name],lw=1)
 if len(y)>=3:axes[1].plot(x[2:],np.convolve(y,np.ones(3)/3,mode='valid'),label=labels[name],color=colors[name],lw=2)
axes[0].set(xlabel='Cumulative training samples',ylabel='Training action loss',title='Loss vs samples')
axes[1].set(xlabel='Optimizer updates',ylabel='Training action loss',title='Loss vs updates')
a=summary['throughput']['baseline_first_15_minutes']['samples_per_second'];b=summary['throughput']['short_6gpu_bs24']['samples_per_second']
axes[2].bar(['7 GPUs\nbatch 84','6 GPUs\nbatch 24'],[a,b],color=[colors['bs84'],colors['bs24']],width=.55)
for i,v in enumerate([a,b]):axes[2].text(i,v+.2,f'{v:.2f}',ha='center')
axes[2].set(ylabel='Samples / second',title='Measured end-to-end throughput',ylim=(0,max(a,b)*1.2))
for ax in axes:
 ax.grid(alpha=.2,axis='y');ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False);axes[1].legend(frameon=False)
fig.suptitle('Same model, dataset, normalization and per-update learning-rate schedule',fontsize=11)
fig.tight_layout()
fig.savefig(root/'comparison.png',dpi=180)
