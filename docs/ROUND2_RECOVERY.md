# Controlled experiment v9: recovery without new training

`results/round2-v9/queue.json` contains the saved registry and exact-version
submission receipts. Kaggle outputs are private; authenticate through the local
Kaggle SDK configuration or environment, never credentials in Git. Retrieve the
pinned versions; do not push kernels to recover completed work.

On a fresh checkout with the documented runner dependencies installed:

```sh
python scripts/controlled_experiment.py restore
python scripts/retrieve_controlled_artifacts.py
```

`restore` refuses to overwrite an existing private registry. If it already exists,
inspect it and skip restore. The retrieval command has no submission operation;
it downloads each terminal attempt and verifies saved source, nonce, receipt,
report checksum and every declared output checksum. `best.pt`, `last.pt`, history,
execution logs, per-image and patient validation predictions remain under ignored
`artifacts/runner/round2-v9/<run>/attempt-<n>/`.

The input cache remains the original private version
`saisathwikregula/trustretina-full-v5-inputs-4f23af6d/1`. Do not rebuild it. Original
validation baselines require their existing full-v5 receipts/artifacts; use the
historical recovery workflow if these are absent. Once both generations are
locally restored and scientific dependencies are installed:

```sh
python scripts/controlled_experiment.py compare
python scripts/report_controlled_results.py
```

Reporting refuses an incomplete matrix by default. `--allow-incomplete` is for an
explicit partial report and never selects a winner. It rehashes artifacts,
recomputes image/patient metrics and confusion counts, reconstructs mean-eye
probabilities and verifies identical patient/label/component alignment with v5.
Resumed histories must preserve the full prior prefix, continue at the next epoch
and use the saved next learning rate.

Source/configuration/protocol hashes are immutable for this experiment. The frozen
protocol's original “approval pending” wording records its prospective state;
execution authorization is the separately saved user approval bound to its plan.
Changing that document would invalidate running/resumable compatibility.

Private checkpoint retention depends on the Kaggle account and saved kernel
versions. These are recoverable private research artifacts, not public model
hosting. Keep local backups where permitted. No new training is authorized by this
recovery document. Never re-create or re-submit a completed run.
