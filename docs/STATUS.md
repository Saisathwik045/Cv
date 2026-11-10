# V14 CPU readiness complete; no GPU execution authorized

Merged PR26/main `f722cb51167b1694802ffdbbe297866b7e8db2a9` verified.
[Readiness/identities/commands](../results/candidate-v14/CPU_READINESS.md),
[actual evidence](../results/candidate-v14/readiness.json),
[worker contract](CANDIDATE_READY_V14.md), [frozen queue](../results/candidate-v14/frozen-plan.json).

One private GPU-disabled CPU job completed:
`saisathwikregula/trustretina-v14-inputs-c057c643/1`,276.103s against3,600s cap.
All1,750 native validation/calibration raw hashes passed; only missing320 tensors
were constructed. TRAIN224/TRAIN320 and existing224 validation/calibration caches
were recovered/hash-verified and reused. Historical final excluded. Actual role
patients/groups: TRAIN2275/2240, validation525/518, calibration350/331; all-eight
support and patient/containment separation passed. No calibration labels in the
new GPU input bundle. Separate original labeled metadata remains private.

Storage: free private /tmp scratch, mode0700, capacity/write tested, enough for
all3 possible sessions' best/last checkpoints plus512MiB margin. No verified
artifact deleted; pinned existing/new Kaggle outputs are durable recovery copies.
Scratch can disappear with the environment; use candidate_recover.py, not a new
preparation job. Exactly one CPU submission used; no further preparation session.

Worker/registry enforce two fresh-ImageNet ConvNeXt ratio-BCE fits,320 then224,
frozen v4 TRAIN/validation roles, max25 total cosine epochs/patience6, TRAIN-only
weights, validation patient macro AUROC selection, strict checkpoint/RNG resume,
mean-eye probabilities, fixed p≥.5 and label-free calibration inference only after
checkpoint freeze. No temperature/threshold/policy fit, demo replacement or full-
cohort refit. Four-hour free GPU cap14,400s/max3sessions retained; zero GPU attempts.
No actual v14 GPU/model inference has run. REVIEW REQUIRED remains default.

BRSET public access requirements checked: PhysioNet credentialing, required CITI
training and personal DUA signature are needed; no restricted metadata entitlement
or files verified, no agreement submitted. Exact links are in CPU_READINESS.md.
No invented label mapping. External confirmation/all-eight cohort readiness stays
blocked; a future development-fit approval must acknowledge that separate limit.

Current next action: review the source/input/config/dependency hashes and grant a
separate exact two-fit GPU approval only if desired; full free quota/resource/
accounting is rechecked immediately before submission. Disabled approval template
and working gated commands are committed. No background monitoring is active.
Nine local guard tests pass; two Torch fixture tests await/are recorded by CPU CI.
[Prior v13 results and harms](../results/resolution-v13/DISCUSSION.md) and
[historical planning](history/STATUS-through-v14-planning.md) remain unchanged.
