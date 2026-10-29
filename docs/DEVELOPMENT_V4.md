# Fixed-schedule review completion and contained development release v4

PR #10 is merged in main `9943135`; its final CI passed. Its blocked real-data
status remains the historical v3 result. This revision changes the duplicate
containment/release protocol, not the original architecture or five backbones.

## Recover completed work and bound remaining execution

The original source/specification defined 1,536 scheduled comparisons. Its timeout
retained 422 detailed results but lost the full content cache/selection file.
Reconstruction therefore reads RGB pixels once to reproduce the original foreground
ranking. It does not reparse the workbook or repeat the completed 7,000-image hash
validation. That validation is reused from the verified, immutable dataset-version-2
report. Selected files are hash-checked when subsequently used for new comparisons.

The reconstruction checkpoints every 250 content vectors, then saves the complete
numeric cache, candidate scores, catalog and selection before detailed work.
The first 422 recovered schedule entries, endpoints and content scores must match
the original results. An environment/identity mismatch blocks continuation. The
remaining 1,114 entries are divided into 384/384/346-comparison batches. Each result
is flushed and fsynced immediately; repeated completed indices and candidate
expansion are rejected. Recovered results are never recomputed.

Additional remote CPU execution has a 7,200-second total limit. The accepted
requests reserve 1,800 + 900 + 900 + 900 = 4,500 seconds of hard session timeouts;
actual stage times and receipts are recorded separately. A rejected oversized
upload had no run, confirmed through API reconciliation; its replacement references
the prior kernel output instead of embedding another copy. No duplicate execution
or GPU smoke is submitted. Private job IDs, source versions, nonces and artifact
hashes are retained before/after polling.

## Evidence and explicit uncertainty containment

The same fixed v3 image method and numeric tolerances are retained: cropped retinal
content, contrast-enhanced 512-pixel SIFT detail, mutual matches, RANSAC geometry,
spatial coverage, foreground SSIM and normalized photometric error. The original
13 controls and validated package/method contract are reused. No thresholds or
seeds are selected from class support, model results or attractive split sizes.

All 14 named cases were rendered from hash-verified original images and technically
reviewed. Some are overexposed/low-information; some differ in visible structure;
two have hundreds of geometric correspondences but fail the strict photometric
criterion. All 14 retain an **unresolved_contained** decision, with a separate
case-specific visual/numeric reason. This is targeted uncertainty containment, not
confirmation of identical photographs, patients or clinical findings. Differing
patient-derived label vectors do not establish annotation errors. No labels or
original images are altered.

Policy `scheduled-content-containment-v4` includes supported-copy links and explicit
containment links for independently flagged material uncertainty within the fixed
review schedule, including the 14 named cases. Connected patient IDs are indivisible;
both eyes always follow their patient. Non-triggering and unscheduled weak dHash
matches remain sensitivity records, not automatic links or exclusions. Prior
supported false-positive decisions are inherited from checksum-bound adjudication,
not hardcoded patient identities. There is no blanket quarantine.

The policy preserves every patient's own label vector, including disagreements
inside a contained group. Reports include disagreeing-edge counts and mixed-label
components; no majority vote or inferred correction is permitted. If these
constraints make the prespecified partitions/class support infeasible, release
stops rather than weakening evidence or changing seeds.

## Release and calibration contract

Use seed 2026, .65/.15/.10/.10 targets and the existing group-aware rarest-label /
whole-component support-repair allocator. Report positive/negative patients and
positive/negative supplying groups for all eight labels. Mixed groups can supply
both outcomes and are reported explicitly; counts are feasibility indicators,
not independent-binomial or power claims. Require five positive and five negative
patients/groups per role and at least 50 calibration components.

Release requires exact completion of the fixed schedule, verified stage/source
receipts, reusable original validation, completed named-case review, no connected
constraint crossing, class support and the CPU cap. New immutable role manifests,
policy, configuration/provenance digests and a final complete marker are written
outside Git; historical releases and blocked diagnostic manifests are preserved.
Passing means this bounded assay is complete and known material uncertainty is
contained. It never means every possible duplicate or shared clinical identity
has been ruled out. Unselected weak candidates and unknown transformations remain
explicit residual risks for sensitivity analysis.

Keep one shared patient-level temperature on the whole calibration partition,
with no class-specific fitting or internal calibration-policy subdivision. Nothing
is fitted during duplicate review or the execution pilot. No clinical risk budget
or acceptance cutoff is inferred from sparse classes; unsupported triage remains
REVIEW REQUIRED. Confidence, MC Dropout, quality, calibration and triage stay in the
architecture. Sparse-class counts, cluster dependence and wide uncertainty remain
limitations even when support floors pass.

## The sole authorized GPU pilot

Only a passing v4 release can construct the pilot request. Select whole training
and validation groups by fixed seed/content-derived order: at most 128 training
patients and 64 validation patients, both eyes retained. No final/calibration rows
are packaged. Root discovery searches one allowed training filename and never
opens data.xlsx or final manifests. The worker reads only hash-verified permitted
training/validation image rows; final images and labels are not accessed.

The pilot uses EfficientNet-B0, shared 224-pixel preprocessing, random initialization
explicitly labeled execution-only, BCEWithLogitsLoss, AdamW 1e-4, batch 16 and exactly
two epochs. The full five-backbone ImageNet plan is unchanged. There is no model
selection or calibration/policy fit. Validation predictions check execution and
strict reload only, not comparative or final performance.

After epoch one, save model/optimizer, progress, Python/NumPy/Torch/CUDA RNG and loader
RNG with full split/config/preprocessing provenance. Strictly reload into a new model
and optimizer, verify exact state/RNG and validation predictions, then train the
remaining epoch from that checkpoint. Final checkpoint/reload and bounded optimizer
step counts are verified. This performs no extra training epoch for comparison.
Logs, validation predictions and checkpoints are private, checksum-verified execution
evidence. One persisted GPU submission is permitted; ambiguous/error submissions
must be reconciled, not blindly repeated. No paid services or full five-model runs.

## Commands

```sh
trustretina check-contained-development --config configs/development-v4.toml
python -m unittest discover -s tests -v
```

`scripts/finalize_review_v4.py --help` describes the checksum-bound offline
finalizer. It consumes pinned cache/batch receipts, prior adjudication, named-case
technical review and CPU ledger. It writes diagnostics on failure and can publish
a new private frozen release/pilot request only when all readiness checks pass.
Actual counts, stage times, pilot outcome and remaining limitations belong in
STATUS.md after observed execution; software tests alone establish none of them.
