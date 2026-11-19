# Guided cloud demonstration and local readiness

Use the existing cloud workspace; no Mac installation or model transfer is needed.
This adds two dashboard workspaces without changing the original inference
implementation, model, preprocessing, calibration, thresholds or research outputs.

## Guided synthetic demo

1. Open **Guided synthetic demo**. Both visibly labelled geometric images are
   generated in memory. They are not retinal photographs, patient data or new
   scientific examples. There is no disease ground truth and no medical meaning.
2. Select **Run guided CPU demo**. The trusted original inference-v8 loader verifies
   the bundle, loads the original model on CPU and predicts once. Its existing
   20 trained-head passes and an initial D-target Grad-CAM for both images run.
3. Inspect eight raw/calibrated probabilities in N,D,G,C,A,H,M,O order and
   **REVIEW REQUIRED**. Confidence, variance and quality remain experimental signals,
   not clinical reliability percentages or quality grades.
4. **Download synthetic demo JSON** explicitly records generated non-patient input
   context and `research_evaluation=false` around the unchanged backend prediction.
   Keep any downloaded operational output private; it is not a V15 result file.
5. Change the synthetic explanation target and select **Update synthetic explanation**.
   Selecting a new target alone runs no inference and does not relabel an old map.
   A heatmap illustrates sensitivity, not a verified lesion or prediction correctness.
6. **Replace synthetic left image** toggles a different generated left fixture and
   clears old results/explanations. The right fixture stays the same. Another model
   prediction requires another explicit Run click.
7. **Reset synthetic demonstration** clears guided results, explanations and the
   replacement choice, returning to the original generated pair. The original
   pair previews remain because they are generated, not uploaded. Other manual-page
   state is separate. The sidebar reset clears the entire app session as before.

The only runtime backend is the original frozen epoch-2 ConvNeXt-Tiny at 224 px,
T=0.7226021421263825, fixed p>=0.5. No fake fallback, V14/V15 substitution,
calibration fitting, training, dataset access or automatic recovery is introduced.
Missing/invalid assets produce a safe error without invented probabilities/maps.
Navigating, changing a target, exporting or rerunning the page does not itself
repeat prediction. The model cache holds the service, not input bytes or results.
Session reset is not a secure-erasure guarantee and cannot revoke downloaded copies.

## Cloud readiness

Select **Run local readiness checks**. This manual, on-demand snapshot checks:

| Check | Evidence | Boundary |
|---|---|---|
| Python and packages | Supported Python minor and exact installed app-lock versions | Version metadata does not prove native binary imports or model operation |
| Original bundle | All five SHA256 checks and frozen contract | Hashing reads opaque model bytes; no checkpoint deserialization |
| Public V15 evidence | Aggregate hash and four reviewed figure hashes | No private rows, recomputed scientific metrics or independent validation |
| Loopback health | Fixed `127.0.0.1` health URL, two-second timeout, at most 64 bytes | A generic health reply does not authenticate process identity or an external route |
| Configuration | Loopback-only binding, CORS and XSRF protection enabled | Not a penetration test, authentication layer or public-deployment approval |
| Free disk | Current local capacity | Not a RAM, throughput or future-availability guarantee |

The health probe disables proxy lookup and redirects, validates the local port,
and never requests user-entered/external URLs. Responses/errors are summarized
without printing credential values, private paths or third-party exception text.
Package-version mismatches and missing/corrupt assets fail the relevant check.

Even when every local check passes, the report states:

- Checkpoint loading and inference: **NOT RUN by this readiness check**.
- External viewing route: **NOT VERIFIED**.
- Uptime: **NOT GUARANTEED**.
- Live demonstration readiness: **CONDITIONAL**; failed local checks make it BLOCKED.
- Clinical validation: **false**; every prediction remains REVIEW REQUIRED.

Use Guided synthetic demo to observe current model operation. Its generated-image
values cannot be used to judge medical performance. Downloading the local readiness
JSON creates a timestamped software snapshot, not a certificate of safety, uptime
or independent clinical validation. Recheck immediately before presenting.

## Starting and viewing inside the cloud

An operator can use the existing documented app environment and an already approved,
hash-verified private bundle:

```sh
TRUSTRETINA_BUNDLE=/absolute/private/inference-v8 \
  CUDA_VISIBLE_DEVICES='' python scripts/run_demo.py
```

Use the actual approved private directory in the operator's environment; do not
commit it or credentials. Keep loopback binding and CORS/XSRF protection enabled,
and usage telemetry disabled as configured. No new public port, tunnel, cloud provider or
paid resource is provisioned by this feature. Missing artifacts are not downloaded.

Inside a supported cloud Desktop viewer, open the cloud browser at
`http://127.0.0.1:8501`. This is a workspace-local address. Opening the same address
in a laptop browser points to that laptop, not the cloud machine. Tool/browser
verification inside the workspace is not evidence that the presenting device can
reach it. Rehearse the actual viewer path separately; see
[existing access limitations](DEMO_V8.md).

The process can stop when the environment pauses or shuts down. A working
snapshot is not permanent hosting, a durable archive or background monitoring.
Preserve the separately authorized backup plan; no private upload or transfer
is performed here. The old Mac-specific work is optional and not a dependency.

## Academic walkthrough

Show Cloud readiness and explain each limit. Run the clearly labelled generated
pair, inspect eight outputs/REVIEW REQUIRED and explain one target. Export the
synthetic JSON, replace the fixture and reset. Then open **Research study & viva**
for the recorded V15 results and negative findings; these are distinct from both
the live original model and the generated-image smoke test. Do not compare their
numbers as clinical evidence or upload private patient cases for presentation.

## Software checks

```sh
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -p test_cloud_demo.py -v
```

UI tests use mocked backend responses and generated images, never private weights
or records. They check explicit actions, original-identity refusal, CPU selection,
exports, stale-target/output clearing and model-free readiness. HTTP tests use
mock responses and reject invalid ports/redirects. These fixtures are software
evidence, not a real-model run, viewing-route test or medical evaluation.
