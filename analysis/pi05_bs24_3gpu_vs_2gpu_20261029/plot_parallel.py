from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/'comparison.json').read_text())
curves=json.loads((root/'curves.json').read_text())
colors={6:'#3468a3',3:'#d37b28',2:'#49986b'}
common_step=min(run['metrics'][-1]['step'] for run in curves.values())
fig,axs=plt.subplots(1,3,figsize=(14,4.3))
for n in (6,3,2):
 rows=[row for row in curves[str(n)]['metrics'] if row['step']<=common_step]
 x=np.array([r['step'] for r in rows]);y=np.array([r['metrics']['loss/action'] for r in rows])
 axs[0].plot(x,y,color=colors[n],alpha=.25,lw=1)
 axs[0].plot(x[2:],np.convolve(y,np.ones(3)/3,mode='valid'),color=colors[n],lw=2,label=f'{n} GPUs x {24//n} samples')
axs[0].set(xlabel='Optimizer updates (global batch 24)',ylabel='Training action loss',title='Aligned early training curves')
axs[0].legend(frameon=False)
rates={n:d['throughput_all_steady'][str(n)] for n in (6,3,2)}
for ax,key,title,ylabel in [(axs[1],'samples_per_second','End-to-end throughput','Samples / second'),
                            (axs[2],'samples_per_second_per_gpu','Throughput per GPU','Samples / second / GPU')]:
 values=[rates[n][key] for n in (6,3,2)]
 ax.bar([f'{n} GPUs\nlocal batch {24//n}' for n in (6,3,2)],values,color=[colors[n] for n in (6,3,2)],width=.6)
 for i,value in enumerate(values):ax.text(i,value+max(values)*.025,f'{value:.2f}',ha='center')
 ax.set(title=title,ylabel=ylabel,ylim=(0,max(values)*1.17))
for ax in axs:
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
fig.suptitle('Same global batch, model, dataset, state, loss and learning-rate schedule\n3-GPU and 2-GPU runs were concurrent; 6-GPU reference was measured earlier',fontsize=11)
fig.tight_layout();fig.savefig(root/'comparison.png',dpi=180)
