# V13 completed-study recovery

All six GPU sessions are complete. **Do not submit, resume training, prepare inputs,
change approvals or launch another driver.** The six-session authorization is
exhausted even though the charged counter was below its six-hour cap. Recovery and
report reproduction below are read-only remote retrieval plus CPU analysis.

`queue.json` contains the exact final registry and submission receipts;
`execution-provenance.json` contains settled accounting, source/output/checkpoint
hashes and pinned private versions. Every best/last checkpoint, history,
aligned eye/patient prediction file and provider log was retrieved. Logs have a
locally computed SHA256 after pinned-version retrieval; worker-declared output
hashes cover the checkpoint/prediction/history files. Strict best reload happened
inside all six real fits. No GPU checkpoint-resume session was needed or claimed.

The approved identities remain:

- Plan: `0fa597609d0390e7414865ed9507cd97e214c99945abd4c1cc095a9924b1cf2a`
- Worker/source: `e8a8df1bbca95b14109f4e16ca31080e9dd9afe4746d42af5e228b96c0a5e80c`
- Configuration: `fcc8c73b752551a5d5b860b3d134e2828b78ea7a4a5aa41c48549b3a717ae0b4`

| Run | Existing private Kaggle version |
|---|---|
| fold0_res224 | saisathwikregula/trustretina-res-v13-f0-224-fb0c42fb/1 |
| fold0_res320 | saisathwikregula/trustretina-res-v13-f0-320-e7853320/1 |
| fold1_res224 | saisathwikregula/trustretina-res-v13-f1-224-19e7d4a3/1 |
| fold1_res320 | saisathwikregula/trustretina-res-v13-f1-320-597875b0/1 |
| fold2_res224 | saisathwikregula/trustretina-res-v13-f2-224-beb4e463/1 |
| fold2_res320 | saisathwikregula/trustretina-res-v13-f2-320-2958393a/1 |

Preparation is already complete and retained in private version
`saisathwikregula/trustretina-res-v13-inputs-767673ae/1`.
`restore_resolution_state.py` downloads its existing actual inputs and restores
exact receipts/folds; it does not rebuild images, recreate splits or submit jobs.
Private version retention and access depend on the owner's Kaggle account.
Do not delete those versions. Allow at least12GB for a fresh full local recovery.

For a **fresh checkout without private files**, use Python3.12 and securely configure
Kaggle's local credentials through its normal configuration; never put tokens in
shell history, Git or this document. Then:

```sh
python3.12 -m venv .venv-v13-analysis
. .venv-v13-analysis/bin/activate
python -m pip install -r requirements/resolution-v13-analysis.lock
python -m pip check
python scripts/restore_resolution_state.py
python scripts/resolution_experiment.py retrieve
python scripts/audit_resolution_execution.py
python scripts/report_resolution_results.py --bootstrap-batch 2000
python scripts/plot_resolution_results.py
python scripts/render_resolution_discussion.py
```

The lock records the actual CPython3.12.14 Linux CPU environment used for this
report, without Torch. Dependency consistency passed; a second clean install was
not repeated. The restoration/retrieval code was used on this study's actual pinned
inputs and outputs. Use the existing environment/private files when present rather
than downloading everything again. Keep all patient artifacts under ignored
`artifacts/`, with private filesystem permissions.

The frozen reporter refuses an incomplete matrix. Bootstrap state is checkpointed
in private `experiment/analysis-bootstrap.json`; the original CPU process vanished
after1,580 saved replicates without a retained traceback. It resumed the identical
source-bound, seed2026 sequence and finished exactly2,000. Logs of both CPU attempts
were preserved privately; no scientific correction or GPU rerun occurred.

`record_resolution_execution.py` regenerates accounting provenance only when the
original private `final-quota.json` and final remote inventory are present. Do not
replace those with today's quota or invent past accounting. The committed
provenance preserves the verified settled counters for a fresh task. Aggregate
reproduction does not need training authorization or a new quota reservation.

The original demo bundle, historical final payload, calibration and exploratory
thresholds are unnecessary for recovery and must remain unopened/unchanged.
No process monitoring continues after this task closes. Further experiments,
refitting or model promotion need a separate scoped decision.
