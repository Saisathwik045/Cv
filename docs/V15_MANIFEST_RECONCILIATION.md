# V15 original V4 manifest reconciliation

Software-only continuation from PR #30/main
`6c33e8c`. No real derived rows, patient records, labels or original manifests
were accessed. The previously recovered 36 V14 analysis artifacts are not opened
by this utility. Nothing here authorizes recovery, fitting, inference, jobs,
historical final access or demo changes. Existing V4/V14/V15 protocols and the
V15 loader's hash/role guards remain unchanged.

## Publisher and lossless ancestry

`src/trustretina/data/contained_release.py:publish_release` (introduced/finalized
in commits `36929fe`/`5b9b737`) writes each original role as:

```python
json.dumps({"schema": "phase2_eye_rows_v1",
            "protocol": "development-content-containment-v4",
            "role": role, "rows": rows}, sort_keys=True)
```

The row keys are exactly `patient_id`, `eye`, `labels`, `component_id`, `path`,
`raw_sha256`. Key ordering is recursive lexicographic ordering. Defaults are
ASCII escaping, comma followed by a space, colon followed by a space, no indent
and no terminal newline. UTF-8 encoding produces the original bytes (ASCII after
escaping). There is no class-order field in the original object; label positions
are N,D,G,C,A,H,M,O. Adding such a field would change the original hash.

The publisher traverses partition patient IDs in their saved allocator order,
then `sorted(eyes.items())`: left before right. Patient order is **not newly
sorted**. `full_inputs.py` saves the entire unmodified role row list, using
`json.dumps(rows, sort_keys=True)`. `reliability_inputs.py` checks validation rows
against the older cache and saves validation/calibration lists the same way.
`resolution_prepare_runtime.py` uses `shutil.copyfile` for TRAIN rows. Thus no
inverse preprocessing, path conversion, label inference or partition generation
is needed. Retain the exact source list order; output SHA256 is the definitive
check of that order and every field. A different order fails rather than being
repaired or searched for. Containment IDs are canonical hashes of sorted member
patient IDs; mixed-label components preserve every patient's own labels.

## Original expected SHA256s

Authority: `results/candidate-v14/frozen-plan.json`,
`registry.role_manifest_sha256`. Frozen-plan SHA256 remains
`43351a86118e6465a0e24939faddecd9b09d8c8af49d1122c8f0f47f197749e9`.

| Original file | Expected SHA256 |
|---|---|
| train.json | `082ede51da1698cf44ac904e6b22a4467765892ad6371fc8fa04c48e148c2775` |
| validation.json | `109edbb890b030695985682ea413d070b724d7469b28aaabc8a19f49cdb71acb` |
| calibration.json | `91fd16b7d6693eee7d98e74335910583fdb3312309d2198cedeeb8bd21d9e969` |

## Exact derived dependencies (not original manifests)

| Role | Authenticated owner/version | Exact remote object | Source SHA256 |
|---|---|---|---|
| TRAIN | `saisathwikregula/trustretina-res-v13-inputs-767673ae/1` | `prepared/train-rows.json` | `2656e5b7403d90123c54e5c493988f1fd889003e8424cec3ebdf33ff016a2885` |
| validation | `saisathwikregula/trustretina-phase6-inputs-0419699f/1` | `prepared/validation-rows.json` | `5e1dd5d80245d9046cf0c7552ce37f66dc4347e2fcbe6e9e316a5e774214acec` |
| calibration | `saisathwikregula/trustretina-phase6-inputs-0419699f/1` | `prepared/calibration-rows.json` | `4e7a8293567d3dcd83705065a87e8971c64c9da18840b7c4bc657fec53502009` |

Owner/version authority: `results/resolution-v13/preparation-receipt.json` and
`results/reliability-v6/run-registry.json` input receipts/stage. TRAIN source hash:
`results/resolution-v13/queue.json`, `registry.cache.rows_sha256`, corroborated by
`prerequisites.json` and the V12 audit. Other source hashes:
V6 `registry.stages.inputs.artifact_hashes`. These source digests authenticate
input lists only; they never substitute for original-manifest digests.

## Working software commands

```sh
.venv/bin/python -B scripts/reconcile_v15_manifests.py plan
.venv/bin/python -B -m unittest discover -s tests -p test_v15_reconciliation.py -v
```

`plan` reads only explicit public source/provenance records. Its canonical
`plan_sha256` binds all records, source code, exact source identities, expected
outputs, class order and counts. It imports no Kaggle SDK and performs no network
operation. Review a fresh plan after any code change.

The offline `reconcile` command exists for a **later separately authorized** use:

```sh
.venv/bin/python -B scripts/reconcile_v15_manifests.py reconcile \
  --input-root /approved/private/derived-rows \
  --output /approved/private/reconciliation-attempt \
  --approval-file /approved/private/reconciliation-approval.json
```

Paths illustrate future approved locations, not current permission. Local inputs
are exactly `train-rows.json`, `validation-rows.json`, `calibration-rows.json`.
No recursive discovery or downloader is implemented here. The existing V14
artifact recovery tool's version/file allowlist is deliberately unchanged; it
cannot be used to fetch these new dependencies without separately reviewed work.

Both input root and output parent must already be user-owned 0700 directories
outside the repository, with no symlink components. Input files must be owned
regular 0600 files. Output must be fresh; output/staging directories use0700 and
files use0600 even under permissive umask. Approvals must come from genuine user
instructions, never from the utility. The direct core API is available for tests;
operators must use the approval-gated entry point for private data.

All three input byte hashes pass before parsing any row. No labels, role
assignments, patient IDs or containment groups are regenerated. Counts are
2,275/525/350 patients. Duplicate patients/images, incomplete/misordered eyes,
invalid or disagreeing labels, invalid component membership and cross-role
leakage block acceptance. Class order is checked in the plan; source/output
hashes authenticate its otherwise unannotated vector positions.

All three reconstructed candidate byte hashes pass before any candidate file is
written. Staged files are rehashed from disk before publication. One rename publishes `manifests/` containing the three verified files.
A fresh attempt directory contains `reconciliation.json`: only role hashes,
plan digest, status and fixed diagnostic codes, never rows, paths or exception
messages. Mismatches preserve input bytes in place and diagnostic hashes, without
saving incorrect reconstructed records. An I/O interruption may leave a private
`.verified/` staging directory; failed writes/readback may leave incomplete or corrupt files. Never consume that
staging directory; require the published `manifests/` directory and independently
rehash all three files. A verified diagnostic alone is insufficient if interruption
prevented publication. Existing attempts are never overwritten or auto-resumed.
The unchanged V15 loader still performs its own original hash/alignment checks.

## Separate authorization required before real use

A new explicit user instruction must authorize BOTH:

1. Activation of Kaggle credentials as `saisathwikregula` and selective recovery
   of **only the three exact owner/version/object/SHA256 dependencies above**,
   including their private patient IDs and original patient-derived labels, into
   named approved durable private storage. Review transfer limits, download
   hosts and selective transport first; do not fetch whole outputs or logs.
2. CPU-only reconstruction/reconciliation of those three local files into a
   named fresh private output, under the reviewed code and plan SHA256, including
   access to their patient/eye/label/component records. Preserve originals and
   diagnostics; stop on any input or original output hash mismatch.

The offline approval record requires `approved=true`,
`scope=v15_private_original_manifest_reconciliation`, the exact `plan_sha256`,
absolute `input_root` and `output`, a genuine nonempty `user_approval_reference`,
and `calibration_fitting_authorized=false`. This is an operator-enforced record,
not a cryptographic approval service, and does not authorize credential use or
recovery on its own. No real approval record is generated by this task.

Recovery/reconciliation approval does not authorize V15 fitting. Fitting needs its
own reviewed real analysis manifest, original/derived role alignment and separate
V15 analysis authorization. No historical final, model inference, training,
calibration fitting, Kaggle job, publication of patient files or demo change is
included in either reconciliation permission.

## Readiness and limits

The lossless method is technically implemented and synthetic-tested, including
byte identity against the actual V4 publisher. It is not yet empirically proven
for the real files. Synthetic tests cannot establish remote retention/access,
real input byte availability, actual expected-hash reproduction, original labels'
clinical validity, unknown leakage, or V15 prediction/quality alignment. Only a
separately authorized exact hash match can establish real reconstruction success.
No semantically equivalent but byte-different output is acceptable; no search,
repartitioning, relabeling or relaxed hash guard is a fallback.
