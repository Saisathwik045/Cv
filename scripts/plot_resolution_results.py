#!/usr/bin/env python3
"""Plots from complete verified v13 aggregates only; no training or fitting."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/resolution-v13';p=json.loads((out/'comparison.json').read_text())
    if len(p['folds'])!=3:raise ValueError('Complete paired matrix required')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/resolution-v13/mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    arms=('res224','res320');fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,('macro_auroc','macro_auprc_average_precision','macro_f1'),('Outer macro AUROC','Outer macro AP','Outer macro F1, p≥.5')):
        for f in p['folds']:ax.plot([0,1],[f['arms'][a][key] for a in arms],marker='o',label=f"Fold {f['outer_fold']}")
        ax.set_xticks([0,1],['224px','320px']);ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Frozen matched development comparison; no independent confirmation');fig.tight_layout();fig.savefig(out/'paired-resolution-metrics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7));x=np.arange(3)
    for row,label in enumerate(('H','O')):
        for ax,key,title in zip(axes[row],('precision','recall','f1'),('Precision','Recall','F1')):
            for j,arm in enumerate(arms):ax.bar(x+(j-.5)*.32,[f['arms'][arm]['per_class'][label][key] for f in p['folds']],.32,label=arm)
            support='/'.join(str(f['arms']['res224']['per_class'][label]['positive']) for f in p['folds'])
            ax.set_xticks(x,['Fold0','Fold1','Fold2']);ax.set_ylim(0,1);ax.set_title(f'{label}: {title}; positives {support}');ax.legend(fontsize=8)
    fig.suptitle('Prespecified H/O outcomes at fixed p≥.5');fig.tight_layout();fig.savefig(out/'H-O-tradeoffs.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4));classes=list('NDGCAHMO');x=np.arange(8)
    for ax,key,title in zip(axes,('fp','fn'),('False-positive label decisions','Missed positive label decisions')):
        for j,arm in enumerate(arms):ax.bar(x+(j-.5)*.36,[next(r[key] for r in p['secondary_pooled'][arm]['confusion'] if r['label']==label) for label in classes],.36,label=arm)
        ax.set_xticks(x,classes);ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('Descriptive pooled outer counts; three models per arm');fig.tight_layout();fig.savefig(out/'all-eight-error-counts.png',dpi=150);plt.close(fig)
    print('Three figures generated from complete verified aggregates; no fitting or inference.')


if __name__=='__main__':main()
