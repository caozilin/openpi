"""Assemble301 generated replacements plus1574 immutable nominal tables; optional authorized retirement."""
import argparse,fcntl,json,os,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/media/czl/sata/franka_my_code/openpi')
import sys
sys.path.insert(0,str(ROOT))
from examples.libero.enrich_nominal_joint13_dataset import read,write,sha,feature_stats
REC=ROOT/'experiment_records/2026-11-09_libero_joint13_successmix20_training'
BASE=ROOT/'data/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109'
OUT=ROOT/'data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109'
STAGE=Path('/tmp/libero_joint13_successmix20_replacement_stage_20261109')

def assemble(retire):
 with (REC/'assembly.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  done=read(STAGE/'replacement_stage_completion.json');assert done['complete'] and done['replacements']==301
  req=read(OUT/'replacement_request.json');assert sha(OUT/'replacement_request.json')==sha(STAGE/'replacement_request.json')
  oldmanifest=read(BASE/'conversion_manifest.json');assert sha(BASE/'conversion_manifest.json')==req['nominal_manifest_sha256']
  original=oldmanifest['episodes'];assert len(original)==1875;selected={e['episode_index'] for e in req['entries']}
  assert read(BASE/'meta/joint13_independent_audit.json')['complete']
  receipts={p.stem:read(p) for p in (STAGE/'replacement_receipts').glob('*.json')};assert len(receipts)==301
  for i in selected:
   receipt=receipts[f'{i:06d}'];p=STAGE/f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet';assert sha(p)==receipt['output_sha256']
  required=sum(r['bytes'] for r in receipts.values())
  if not retire and shutil.disk_usage(OUT).free<required+(512<<20):raise RuntimeError('Additional disk space still required; original conversion not removed')
  archive=REC/'retired_nominal_joint13_conversion_metadata'
  if retire:
   assert read(REC/'storage_authorization.json')['allow_retire_untrained_nominal_conversion']
   # Preserve every small metadata/receipt/manifest before deleting only the known Parquet copies.
   for p in BASE.rglob('*'):
    if p.is_file() and p.suffix!='.parquet':
     dest=archive/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True)
     if not dest.exists():shutil.copyfile(p,dest)
     assert sha(p)==sha(dest)
  for row in original:
   i=row['episode_index'];rel=row['output_path'];p=BASE/rel;assert p.is_file() and sha(p)==row['output_sha256'],rel
   if i not in selected:
    dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():os.link(p,dest)
    assert os.stat(p).st_ino==os.stat(dest).st_ino or sha(dest)==row['output_sha256']
  if retire:
   # After each unaffected table has a second hard link, retire the redundant old namespace.
   deleted=[]
   for row in original:
    p=BASE/row['output_path'];deleted.append(dict(path=str(p),sha256=row['output_sha256'],bytes=p.stat().st_size,kept_as_new_nominal_hardlink=row['episode_index'] not in selected));p.unlink()
   write(REC/'nominal_conversion_retirement.json',dict(complete=True,scope='only1875 Parquet in untrained enriched nominal conversion',files=deleted,original_nominal_dataset_untouched=True,tolerance_replay_untouched=True,labels_untouched=True,metadata_archive=str(archive)))
   for name in ['conversion_completion.json','meta/info.json']:
    p=BASE/name
    if p.exists():p.rename(p.with_name(p.name+'.retired'))
   write(BASE/'retirement.json',dict(retired=True,replaced_by=str(OUT),evidence=str(REC/'nominal_conversion_retirement.json'),original_metadata=str(archive)))
  for i in sorted(selected):
   row=receipts[f'{i:06d}'];rel=f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet';dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists() or sha(dest)!=row['output_sha256']:
    tmp=dest.with_suffix('.assembling.parquet');shutil.copyfile(STAGE/rel,tmp);assert sha(tmp)==row['output_sha256'];tmp.replace(dest)
   write(OUT/'replacement_receipts'/f'{i:06d}.json',row)
  meta_source=archive/'meta' if retire else BASE/'meta';meta=OUT/'meta';meta.mkdir(exist_ok=True)
  for name in ['episodes.jsonl','tasks.jsonl','reconstruction_sources.json','tolerance_sources.json','annotation_protocols.json','source_reconstruction_audit.json']:shutil.copyfile(meta_source/name,meta/name)
  original_protocol=read(meta/'annotation_protocols.json');write(meta/'source_annotation_protocols.json',original_protocol)
  write(meta/'annotation_protocols.json',dict(label_provenance=original_protocol,trajectory_counts={'nominal':1574,'tolerance':301},trajectory_types={'0':'nominal','1':'tolerance'},replacement_request_sha256=sha(OUT/'replacement_request.json'),video_policy='selected tolerance H264MP4 decoded pixels, no additional rotation',actions='all unchanged corresponding source nominal actions7'))
  tolerance_sources=read(meta/'tolerance_sources.json')
  for e in tolerance_sources['episodes']:
   i=e['source']['episode_index'];e['trajectory_is_tolerance']=int(i in selected)
   if i in selected:e['observation_source']=next(x['replay'] for x in req['entries'] if x['episode_index']==i)
  write(meta/'tolerance_sources.json',tolerance_sources)
  info=read(meta_source/'info.json');info['joint13_contract']['trajectory_counts']={'nominal':1574,'tolerance':301};info['joint13_contract']['replacement_protocol']='per-task round20% of successful tolerance replay pool';info['joint13_contract']['tolerance_images']='decoded H264MP4 encoded as PNG; already rotated180 once';write(meta/'info.json',info)
  stats=[json.loads(x) for x in (meta_source/'episodes_stats.jsonl').read_text().splitlines()]
  manifest=[]
  for old in original:
   i=old['episode_index']
   if i in selected:
    row=receipts[f'{i:06d}'];s=stats[i]['stats'];s.update(state=row['state_stats'],image=row['image_stats'],wrist_image=row['wrist_image_stats'],trajectory_is_tolerance=feature_stats(np.ones((row['frames'],1),np.int64)))
    manifest.append(dict(episode_index=i,frames=row['frames'],output_path=old['output_path'],output_sha256=row['output_sha256'],trajectory_is_tolerance=1))
   else:manifest.append(dict(episode_index=i,frames=old['frames'],output_path=old['output_path'],output_sha256=old['output_sha256'],trajectory_is_tolerance=0))
  (meta/'episodes_stats.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in stats));write(meta/'replacement_sources.json',req)
  write(OUT/'conversion_manifest.json',dict(episodes=manifest));write(OUT/'conversion_request.json',req)
  write(OUT/'conversion_completion.json',dict(complete=True,episodes=1875,frames=312374,nominal_episodes=1574,tolerance_episodes=301,request_sha256=sha(OUT/'replacement_request.json'),independent_audit_pending=True))
  write(OUT/'conversion_state.json',dict(stage='complete',processed=1875,total=1875,nominal=1574,tolerance=301))
  write(REC/'assembly_completion.json',dict(complete=True,episodes=1875,nominal=1574,tolerance=301,old_conversion_retired=retire,full_independent_audit_pending=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--retire-untrained-nominal-copy',action='store_true');a=p.parse_args();assemble(a.retire_untrained_nominal_copy)
