#!/usr/bin/env python3
"""Check PyPI wheel metadata for locked Mac dependencies; never claims native execution."""
import concurrent.futures,json,sys
from pathlib import Path
import requests
from packaging.tags import cpython_tags,compatible_tags,mac_platforms
from packaging.utils import parse_wheel_filename,canonicalize_name
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.version import Version
from packaging.specifiers import SpecifierSet
root=Path(__file__).resolve().parents[1];pins={}
for path in ['requirements/app-macos-arm64.lock','requirements/artifact-recovery.lock']:
 pins[path]=dict(line.split('==') for line in (root/path).read_text().splitlines() if line and not line.startswith('#'))
allpins=set((n,v) for p in pins.values() for n,v in p.items())
def fetch(pair):
 n,v=pair;r=requests.get(f'https://pypi.org/pypi/{n}/{v}/json',timeout=30);r.raise_for_status();return pair,r.json()
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:metadata=dict(pool.map(fetch,allpins))
import hashlib
report={'lock_sha256':{path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in pins},'scope':'PyPI wheel/Requires-Python/dependency metadata checks from Linux, not macOS execution','target':'macOS 13+ Apple silicon native CPython 3.11/3.12','locks':{}}
for name,packages in pins.items():
 report['locks'][name]={}
 for minor in (11,12):
  platforms=list(mac_platforms((13,0),'arm64'));tags=set(cpython_tags((3,minor),platforms=platforms))|set(compatible_tags((3,minor),interpreter=f'cp3{minor}',platforms=platforms));items={}
  env=default_environment();env.update(sys_platform='darwin',platform_system='Darwin',platform_machine='arm64',python_version=f'3.{minor}',python_full_version=f'3.{minor}.12',extra='')
  for n,v in packages.items():
   data=metadata[n,v];matches=[]
   if data['info']['requires_python'] and Version(f'3.{minor}.12') not in SpecifierSet(data['info']['requires_python']):raise ValueError(f'Unsupported Python requirement: {n}=={v}')
   for f in data['urls']:
    if f['filename'].endswith('.whl') and tags&parse_wheel_filename(f['filename'])[3] and (not f['requires_python'] or Version(f'3.{minor}.12') in SpecifierSet(f['requires_python'])):matches.append({'filename':f['filename'],'sha256':f['digests']['sha256']})
   if not matches:raise ValueError(f'No compatible wheel: {n}=={v} Python3.{minor}')
   for requirement in data['info']['requires_dist'] or []:
    r=Requirement(requirement)
    if r.marker and not r.marker.evaluate(env):continue
    key=canonicalize_name(r.name)
    if key not in packages or Version(packages[key]) not in r.specifier:raise ValueError(f'Missing/conflicting dependency {n}: {r}')
   items[n]={'version':v,'compatible_wheels':matches}
  report['locks'][name][f'3.{minor}']=items
(root/'results/demo-v8/macos-wheel-metadata.json').write_text(json.dumps(report,indent=2))
print('Both exact locks have compatible macOS13 ARM64 wheels and closed dependency constraints for CPython3.11 and3.12.')
