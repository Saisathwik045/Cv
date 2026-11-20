# Original Streamlit demo: measured CPU performance

**Operational synthetic benchmark only. Independent clinical validation remains
incomplete. All outputs remain REVIEW REQUIRED.** The planned development
architecture is implemented; this measurement documents its execution cost, not
retinal accuracy, disease detection, clinical readiness or independent validation.

## Observed result

[Machine-readable aggregate](../results/demo-cpu-performance/measurement.json)
was collected on 2026-11-20T04:50:42.516950+00:00 in one fresh Linux Python process.

| Operation | Observed time | Scope |
|---|---:|---|
| Initial five-file SHA256 verification | 0.511776 s | Before dependency import/model construction |
| Dependency import | 3.466298 s | Torch and trusted inference service, not complete process startup |
| Original model load | 0.806835 s | Includes constructor's repeated bundle verification, model construction, strict weights-only load and state hash |
| First paired prediction | 0.208478 s | First call after load; OS caches not flushed |
| Repeated paired prediction: median | 0.151457 s | Three sequential calls on the same generated pair |
| Repeated paired prediction: range | 0.144690–0.174269 s | Sample range, not an interval estimate or p95 |
| Paired D-target Grad-CAM | 0.174469 s | One explanation for both images, including preparation and overlays |

The prediction timer wraps the existing full `predict` call: PNG decode,
224 px preprocessing, backbone features, sigmoid/mean-eye aggregation, saved
20-pass trained-head MC, technical quality and the frozen shared temperature.
It excludes separate Grad-CAM and Streamlit/browser overhead. The first call
serves as the warm-up for the three repeated calls; no extra predictions ran.
Timing validation/model-state checks are outside the individual call timers.

Process peak RSS before model construction was
**361,304,064 bytes
(344.57 MiB)**;
after measurement it was **811,536,384
bytes (773.94 MiB)**.
These are Linux process-lifetime high-water marks, not current RSS, incremental
model memory, an enforced memory cap or a minimum RAM requirement. Imports,
constructor checkpoint/state copies and Grad-CAM can contribute. The measurements
do not characterize total cloud host memory or concurrent sessions.

## Environment and identity

- Linux x86_64, Python 3.12.14; torch 2.7.1+cpu,
  torchvision 0.22.1+cpu, NumPy 2.2.6,
  Pillow 11.3.0.
- Three logical/affinity CPUs exposed; cgroup `cpu.max=200000 100000` permits 0.2
  CPU-seconds per 0.1-second period (two CPU equivalents), not three dedicated cores.
- Cgroup memory limit: 8,589,934,592 bytes (8 GiB).
- Torch intra-op threads: two; inter-op threads: one. CUDA hidden; networking disabled.
- Original epoch-2 ConvNeXt-Tiny, 224 px, T=0.7226021421263825, p >= 0.5,
  classes N,D,G,C,A,H,M,O; no V14/V15 model substitution.
- Source commit: `72ab6f736cc79ff5fecbb1049e5975e9a760c9c6`.
- Benchmark harness SHA256: `db05851758a97a7510f902433c05d01ba6a1a6e10ea394868f751be6769655e5`.
- Measurement module SHA256: `729dc92852d1cc16d7c11dd35a6a8bdb449ce6d9b4443d134187f9da4652b7c3`.
- Aggregate SHA256: `5830473786b52d697935927db9c39966ac091d16688a9e7074d4980550d6891f`.
- All five original bundle identities verified before/after; model state and eval
  modes unchanged. Exact bundle hashes are in the aggregate. No predictions,
  images, patient records, signed links or credentials are included.

## Reproduce only when authorized

Use the existing Linux CPU environment and an already approved, verified private
original bundle. No download/recovery is performed. The explicit execution flag
loads real weights and runs exactly four synthetic paired predictions plus one
paired explanation; it is not a model-free command. Run in a fresh process:

```sh
umask 077
mkdir -m 700 /absolute/private/new-cpu-measurement
CUDA_VISIBLE_DEVICES='' python -B scripts/benchmark_demo_cpu.py \
  --run-synthetic-cpu \
  --bundle /absolute/private/inference-v8 \
  --output /absolute/private/new-cpu-measurement/measurement.json
```

Replace placeholder paths with an approved existing bundle and a fresh private
output directory outside the checkout. Files are 0600, directories owner-only,
symlink paths rejected and existing output files never overwritten. The Linux-only
CLI has a 180-second deadline and fails without publishing success if an identity,
state or output check fails. It is not wired to dashboard navigation, normal CI
or automatic background execution. CI tests use a mock service, not real weights.
Review aggregate contents before any separate publication; never publish model
files or synthetic prediction arrays. The command deliberately does not claim
native Mac support or benchmark parity.

## Interpretation and limitations

The original backend operated comfortably within this small sequential workload
in this environment. A presenter should still rehearse the actual cloud Desktop
route and both pages. These numbers do not measure browser/upload/download time,
server startup, cold disk caches, contention, simultaneous users, long-run
stability, throughput, other image dimensions or native Mac hardware. No p95,
confidence interval, scalability target or clinical performance conclusion can
be estimated from three repeated timings. No latency optimization or model change
was made, and no timing pass/fail service-level target was defined after seeing
the results. Synthetic shapes have no disease ground truth or medical meaning.

Development architecture completion and operational performance are separate
from independent scientific confirmation. See the
[submission checklist](FINAL_SUBMISSION_CHECKLIST.md) and
[cloud demonstration boundaries](CLOUD_DEMONSTRATION.md).
