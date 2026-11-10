# V14 authorized execution — 320 candidate submitted, 224 reference pending

Merged PR27/main3b07a7634f3e6530d66b2ec41c70d837893992c6 verified.
User authorized the exact frozen two-fit queue under14400 free GPU-quota seconds/
maximum3sessions including failures, overhead and at most one compatible resume.
No paid resources. Plan3e82ea2c70524b2d7348ea4f7df8bc818535f8db95ec6ce7f15923693d4b0a18.
Worker/config/protocol/dependencies and actual six input tensors/ImageNet weights
passed checksum verification. Original CPU preparation and v13 training not repeated.

Preflight remote v14 inventory empty; actual free quota14673.105s, reserved0, paid
disabled. Original cumulative baseline timeUsed6926.895s, refresh2026-11-14T00:00Z
is saved before first submission and must never be reset on interruption.

First job: saisathwikregula/trustretina-v14-fit-320-7bbe5ade/1. Submission accepted;
its exact receipt/source/nonce/reservation7800s is saved. One of3 sessions used.
The224 reference is not submitted; both fits are already authorized, sequentially.
Provider queued/running/completed state must be read; no completion claimed yet.
[Machine-readable registry](../results/candidate-v14/frozen-plan.json),
[preflight](../results/candidate-v14/execution-preflight.json),
[actual input verification](../results/candidate-v14/actual-input-verification.json).

Resume using the existing private registry/approval and exact recorded version.
`candidate_experiment.py advance --submit --approval-file
artifacts/candidate-v14/experiment/approval.json` performs one guarded bounded
advance; poll no faster than180s. Reconcile ambiguity; never blind-resubmit.
Retrieve/hash-verify existing output before any next submission; preserve global
budget baseline/session count. Training/calibration/final roles stay separated.
No temperature/threshold fitting, external evaluation, historical final access or
demo modification. BRSET access/agreements remain a separate user-controlled
blocker. REVIEW REQUIRED remains unchanged. No monitoring is promised after a turn.

[CPU-readiness evidence](../results/candidate-v14/CPU_READINESS.md) and frozen
[CANDIDATE_READY_V14.md](CANDIDATE_READY_V14.md) retain the unchanged protocol.
