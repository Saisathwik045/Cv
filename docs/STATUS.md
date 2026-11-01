# V12 CPU development audit complete — no training authorized

[Consolidated v9–v12 report](../results/development-audit-v12/REPORT.md),
[aggregate evidence](../results/development-audit-v12/audit.json),
[prospective sampling rules](DEVELOPMENT_AUDIT_V12.md), and
[one next-experiment proposal](NEXT_EXPERIMENT_AFTER_V12.md).

PR #22 is already merged at `8e67b82`; CI41309020903 passed all seven jobs.
Its reporting/runner/aggregation/accounting paths were reviewed. The numerical
sqrt-loss AUROC nomination does not establish improvement: paired mean difference
+0.00098 (conditional 95% interval −0.00930 to +0.01058), with macro F1 down .02541.
Keep ratio BCE as development reference; preserve both loss implementations.

The bounded audit used six existing v11 training-only outer prediction pairs and
one pass over 4,550 cached 224px eyes (12 seconds automated wall time). All 2,275
training labels and raw-image references match saved source inventory; source
receipts/cache/prediction hashes pass. Keywords cover only 1,835 patients and are
selection-biased. H has 42/67 multilabel positives and O 275/636. Of 528 covered
O positives, 259 have a literal normal-fundus keyword in exactly one eye; 54/60
covered H positives have H keywords in both eyes. Source consistency is not
clinical annotation verification. No labels were changed.

All 32 fixed TP/FP/FN/TN stratum records were technically inspected: 18 unique
paired inputs / 36 eyes. Correct predictions and errors both include variable
framing, softness and illumination. H true/false positives can share small fields
and wide padding. No sole quality, sparse-support or annotation-error explanation
is established. Original images are not local; native-detail and raw-to-cache
parity checks remain unavailable. Private selection, arrays, keywords, montages
and individual notes stay in `artifacts/audit-v12`; public hashes permit integrity
checks. A report-only correction removes precision/F1 conditioned on true-positive
subgroups; original output and explanation are preserved, without repeating images.

**Next action:** recover/verify native TRAIN-only inputs and confirm H/O annotation
semantics before requesting training approval. The sole proposed follow-up is
three paired containment folds of 224 versus 320px ConvNeXt cosine ratio BCE.
Estimated ceiling 4.23 worker-hours (5.50 with allowance); proposed strict six free
GPU-quota hours / six sessions, including failures/resumes/overhead. No queue is
submitted or new cache created. Increasing resolution is an unconfirmed hypothesis,
not a promised gain. Freeze/test the implementation and obtain separate exact-plan
approval first. Stop if the prerequisites or quota cannot support it.

214 CPU tests ran: 205 passed, nine optional skips; compilation and diff checks
passed. New tests cover fixed component-distinct sampling, missing strata,
threshold/count boundaries, technical padding masks, source-receipt tampering
and positive-only support reporting.

A fresh independent paired-eye/all-eight cohort is not currently secured or
verified. Its labels, rights, overlap exclusion, representative enrollment and
clustering-aware all-class precision/power require prospective planning. The old
final holdout cannot confirm a new development choice. No new data was uploaded.

The demo checkpoint/temperature/normalizer/contract hashes match the prior frozen
bundle. Historical final, calibration, thresholds, raw labels, all five factories
and both losses remain unchanged. No training, new inference, GPU submission,
full-data refit, threshold deployment or background monitoring. REVIEW REQUIRED
remains default. [Previous completed study and recovery](history/STATUS-through-v11-results.md).
