#!/usr/bin/env python3
"""Launch only a loopback/private preview; no hosting or credential access."""
import os
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.execv(sys.executable, [sys.executable, '-m', 'streamlit', 'run', str(root / 'streamlit_app.py'), *sys.argv[1:]])
