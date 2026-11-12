"""Run inside the authorized new server run; repository remains read-only."""
from pathlib import Path
import json,hashlib,shutil
R=Path('/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109');REPO=Path('/home/czl/openpi');OVER=R/'source_overlay'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert not OVER.exists(),'Do not overwrite an existing source overlay'
incoming=json.loads((R/'incoming/incoming_manifest.json').read_text())['sha256']
for name,h in incoming.items():assert sha(R/'incoming'/name)==h,name
protected={str(p.relative_to(REPO)):sha(p) for base in (REPO/'src/openpi',REPO/'scripts') for p in base.rglob('*.py')}
write(R/'repository_source_before.json',dict(repository=str(REPO),sha256=protected))
shutil.copytree(REPO/'src/openpi',OVER/'openpi',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(REPO/'scripts',OVER/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
for name in incoming:
 p=R/'incoming'/name
 if name.startswith('src/openpi/'):dst=OVER/'openpi'/name.removeprefix('src/openpi/')
 else:dst=R/p.name
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
manifest={str(p.relative_to(OVER)):sha(p) for p in OVER.rglob('*.py')}
write(R/'overlay_manifest.json',dict(sha256=manifest,incoming_sha256=incoming,repository_copy_sha256=sha(R/'repository_source_before.json')))
assert all(sha(REPO/name)==h for name,h in protected.items())
write(R/'overlay_preparation.json',dict(complete=True,repository_unchanged=True,files=len(manifest),overlay_manifest_sha256=sha(R/'overlay_manifest.json')))
print(json.dumps(json.loads((R/'overlay_preparation.json').read_text())))
