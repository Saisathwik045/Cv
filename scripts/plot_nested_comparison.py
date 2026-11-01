#!/usr/bin/env python3
"""Figures from verified complete-matrix aggregates; no fitting or prediction access."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/nested-v11';p=json.loads((out/'comparison.json').read_text())
    if len(p['folds'])!=3:raise ValueError('Complete three-fold paired comparison required')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/runner/nested-v11/mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    arms=('ratio_bce','sqrt_ratio_bce');fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,('macro_auroc','macro_auprc_average_precision','macro_f1'),('Outer macro AUROC','Outer macro AP','Outer macro F1 at p≥.5')):
        for f in p['folds']:ax.plot([0,1],[f['arms'][a][key] for a in arms],marker='o',label=f"Fold {f['outer_fold']}")
        ax.set_xticks([0,1],['Ratio BCE','Sqrt ratio BCE']);ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Prespecified paired development comparison; no independent confirmation');fig.tight_layout();fig.savefig(out/'paired-outer-metrics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4));x=np.arange(3)
    for ax,key,title in zip(axes,('precision','recall','f1'),('H precision','H recall','H F1')):
        for j,arm in enumerate(arms):ax.bar(x+(j-.5)*.32,[f['arms'][arm]['per_class']['H'][key] for f in p['folds']],.32,label=arm)
        ax.set_xticks(x,['Fold 0','Fold 1','Fold 2']);ax.set_ylim(0,1);ax.set_title(title);ax.legend(fontsize=7)
    fig.suptitle('H sparse support: '+' / '.join(str(f['arms']['ratio_bce']['per_class']['H']['positive']) for f in p['folds'])+' positive outer patients');fig.tight_layout();fig.savefig(out/'H-outer-tradeoffs.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4));labels=[r['label'] for r in p['secondary_pooled']['ratio_bce']['confusion']];x=np.arange(8)
    for ax,key,title in zip(axes,('fp','fn'),('False-positive label decisions','Missed positive label decisions')):
        for j,arm in enumerate(arms):ax.bar(x+(j-.5)*.36,[r[key] for r in p['secondary_pooled'][arm]['confusion']],.36,label=arm)
        ax.set_xticks(x,labels);ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('Descriptive pooled outer decisions at p≥.5; three models per arm');fig.tight_layout();fig.savefig(out/'all-eight-error-counts.png',dpi=150);plt.close(fig)
    print('Three aggregate comparison figures generated; no new inference or fitting.')


if __name__=='__main__':main()
