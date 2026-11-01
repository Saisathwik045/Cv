#!/usr/bin/env python3
"""Publish permitted tables/curves from the verified historical development audit."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/round2-v9'
r=json.loads((OUT/'development-audit.json').read_text());names=('convnext_tiny','densenet121');classes=['N','D','G','C','A','H','M','O']
lines=['# Second-round development diagnosis','', 'Generated from the checksum-bound audit of existing histories and validation predictions. No fitting or final-evaluation rerun. The historical final aggregate is verified only in a separate private preservation record and is not used here.','', '## Support and retained partition structure','', '| Label | Train + / − patients | Validation + / − patients | Validation + / − supplying components |','|---|---:|---:|---:|']
for c in classes:
 t=r['development_roles']['train']['support'][c];v=r['development_roles']['validation']['support'][c]
 lines.append(f"| {c} | {t['positive_patients']} / {t['negative_patients']} | {v['positive_patients']} / {v['negative_patients']} | {v['positive_components']} / {v['negative_components']} |")
lines+=['','Both eyes remain with their patient; all containment groups retain their existing roles. Validation has 518 components; training has 2,240. Mixed-label groups can supply both outcomes, so supplying counts are not independent-binomial sample sizes. Validation is already development-exposed; H has only 16 positives.','', '## Observed selection and curves','', '| Backbone | Historical loss-selected epoch | Selected AUROC | Best recorded AUROC epoch / score | 25-epoch whole-worker extrapolation |','|---|---:|---:|---:|---:|']
for name in names:
 x=r['training'][name];lines.append(f"| {name} | {x['selected_epoch']} | {x['selected_validation_metrics']['macro_auroc']:.5f} | {x['recorded_max_auroc_epoch']} / {x['recorded_max_auroc']:.5f} | {x['extrapolated_25_epoch_worker_hours']:.3f} h |")
lines+=['','Training loss keeps falling as validation loss deteriorates: evidence against assuming that a longer run alone will help. AUROC and weighted loss favor different epochs. The prospective AUROC-selected baseline aligns the selection objective; its comparison with a cosine arm changes only the LR schedule within this round. Historical v5 results are preserved.','', '![Historical loss and AUROC trajectories](historical-curves.png)','', 'Training image BCE and validation mean-eye patient BCE use different units/predictors; absolute gaps must not be interpreted as directly comparable generalization errors. LR was declared constant 1e-4 in the checksum-bound v5 configuration, with no scheduler.','', '## Per-class validation errors at fixed p ≥ 0.5','', '| Backbone | Label | + / − | Predicted + | AUROC | AP | Precision | Recall | F1 |','|---|---|---:|---:|---:|---:|---:|---:|']
for name in names:
 for c in classes:
  x=r['training'][name]['selected_validation_metrics']['per_class'][c]
  lines.append(f"| {name} | {c} | {x['positive']} / {x['negative']} | {x['predicted_positive']} | {x['auroc']:.4f} | {x['average_precision']:.4f} | {x['precision']:.4f} | {x['recall']:.4f} | {x['f1']:.4f} |")
lines+=['','Unclipped H training positive weight is 32.955; ConvNeXt predicts H positive in 183 validation patients with only 16 positives (11 TP, 172 FP, 5 FN). This motivates a later separate threshold/weighting study, not a claim that annotation labels are wrong or that threshold tuning improves ranking. N/D/O ranking also limits performance. All patient-derived labels remain unchanged.','', '## Proposed controlled queue','', 'Exactly four new fits: both backbones × constant/cosine LR, 25-epoch maximum, patience six on all-eight validation macro AUROC, seed 2026, ImageNet V1, original 224-pixel preprocessing/weighted BCE and fixed p=0.5. Higher resolution, alternative loss, threshold fitting and calibration are outside this approval.','', f"Linear historical worker-time extrapolation: {2*sum(r['training'][n]['extrapolated_25_epoch_worker_hours'] for n in names):.3f} h. Request four GPU-quota hours: max six 30-minute worker sessions plus ten-minute overhead guard per attempt. Failed/ambiguous attempts count; no automatic retries beyond caps. Cold start/accounting can differ from the extrapolation.",'', 'These are development comparisons on a repeatedly used cohort, not evidence of new generalization performance. A new suitable independent evaluation cohort and prospective protocol are required. No demo replacement or ACCEPT threshold follows.']
(OUT/'DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/runner/round2-v9/mpl-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'artifacts/runner/round2-v9/cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(11,7))
for row,name in enumerate(names):
 h=r['training'][name]['history'];epochs=[x['epoch'] for x in h]
 axes[row,0].plot(epochs,[x['training_image_weighted_bce'] for x in h],marker='o',label='Training image weighted BCE')
 axes[row,0].plot(epochs,[x['validation_patient_weighted_bce'] for x in h],marker='o',label='Validation patient weighted BCE')
 axes[row,0].set_title(name+' — distinct loss units');axes[row,0].legend(fontsize=8)
 axes[row,1].plot(epochs,[x['validation_patient_macro_auroc'] for x in h],marker='o');axes[row,1].axvline(r['training'][name]['selected_epoch'],linestyle='--',color='gray',label='Historical loss-selected epoch');axes[row,1].set_title(name+' — validation patient macro AUROC');axes[row,1].legend(fontsize=8)
 for ax in axes[row]:ax.set_xlabel('Completed epoch');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'historical-curves.png',dpi=150);plt.close(fig)
print('Generated development-only diagnosis and historical curves; no new fit.')
