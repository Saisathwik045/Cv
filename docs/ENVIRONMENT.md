# Observed environment — 28 October 2026

Repository inspected: `/workspace/Cv`, initial commit 4cbc1b6, only README/.gitignore.
Default remote branch is main (verified with GitHub connector). No dataset/notebooks.
The other attached repository is an unrelated compiler and was not changed.

| Capability | Actual observation |
| --- | --- |
| CPU | AMD EPYC 9V74; 3 logical CPUs visible; cgroup cpu.max 200000/100000 = 2 CPU quota |
| RAM | Host-visible ~9.7 GiB; container limit 8 GiB; no swap |
| Disk | ~31.45 GiB total, ~29.83 GiB free initially (dynamic) |
| Python | 3.12.14, Linux x86_64 |
| PyTorch | Not installed; no CUDA runtime test possible |
| GPU | No nvidia-smi and no /dev/nvidia devices; usable GPU not verified |
| Packages | NumPy, pandas, Pillow, openpyxl, sklearn and setuptools available; no Kaggle CLI |
| Local Git | Available, author configured; branch creation and commits supported |
| gh | Installed; auth status reports invalid GH_TOKEN; token value not inspected |
| GitHub connector | Repository read and advertised push/admin permission verified; publication outcome recorded in STATUS |
| Shell HTTPS | PyPI, GitHub and Kaggle probes all fail connecting to configured proxy:8080 |
| Network policy | Enforced restricted HTTP allowlist; package/GitHub hosts listed, Kaggle absent; no VPN/TCP grants |
| Credentials | Managed readiness reports no secret bindings/outbound identities/runtime variables; Kaggle env/legacy-file presence checks false |

Host totals do not override cgroup limits. Policy allowance does not prove working
connectivity. The GitHub connector is a separate working network path and is not a
Kaggle API transport. No network workaround or proxy bypass was attempted.

A normal pip install could build the local package but user-site installation was
blocked by read-only /home/agent. Installation succeeded in `.venv` under the
workspace with system-site-packages and no build isolation/downloads. No GPU
packages were installed. Reproduce capability observations with
`trustretina check-env --network`; use `--require-gpu` for a failing readiness gate.
