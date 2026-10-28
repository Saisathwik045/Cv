"""Safe capability reporting without exposing credential values."""
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import urllib.error
import urllib.request


def read_limit(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def probe_url(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return {"reachable": True, "http_status": response.status}
    except urllib.error.HTTPError as exc:
        return {"reachable": True, "http_status": exc.code}
    except (OSError, urllib.error.URLError):
        return {"reachable": False, "reason": "connection_failed"}


def environment_report(network=False):
    disk = shutil.disk_usage(Path.cwd())
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "cpu_logical_visible": os.cpu_count(),
              "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "cpu_quota_raw": read_limit("/sys/fs/cgroup/cpu.max"),
              "memory_limit_bytes_raw": read_limit("/sys/fs/cgroup/memory.max"),
              "host_memory_kib": next((x.split()[1] for x in (read_limit("/proc/meminfo") or "").splitlines()
                                       if x.startswith("MemTotal:")), None),
              "disk_total_bytes": disk.total, "disk_free_bytes": disk.free,
              "git_available": shutil.which("git") is not None,
              "gh_available": shutil.which("gh") is not None,
              "kaggle_cli_available": shutil.which("kaggle") is not None,
              "kaggle_credential_presence_only": {
                  "token_variable": bool(os.getenv("KAGGLE_API_TOKEN")),
                  "legacy_variables": bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")),
                  "token_file": (Path.home() / ".kaggle" / "access_token").is_file(),
                  "legacy_file": (Path(os.getenv("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json").is_file()},
              "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
              "torch_installed": importlib.util.find_spec("torch") is not None,
              "cuda_available": None, "gpu_devices": [], "network_probes": {}}
    if report["torch_installed"]:
        try:
            import torch
            report.update(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                          cuda_available=torch.cuda.is_available())
            if report["cuda_available"]:
                report["gpu_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception as exc:
            report["torch_import_error"] = type(exc).__name__
    if network:
        report["network_probes"] = {url: probe_url(url) for url in
                                    ["https://pypi.org/simple/", "https://github.com/",
                                     "https://www.kaggle.com/api/v1/kernels/list"]}
    return report


def require_gpu(report):
    return report.get("cuda_available") is True and bool(report.get("gpu_devices"))
