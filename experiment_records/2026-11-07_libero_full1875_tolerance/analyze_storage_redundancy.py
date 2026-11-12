from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,datetime
RUN=Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107')
OUT=Path('/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance')
encode=lambda x:json.dumps(x,ensure_ascii=False,indent=2)
def prop_size(key,value,indent):
 s=encode(value)
 return indent+len(json.dumps(key,ensure_ascii=False).encode())+2+len(s.encode())+s.count('\n')*indent+2
paths=list(RUN.rglob('trial.json'));allfiles=[p for p in RUN.rglob('*') if p.is_file()];totalbytes=sum(p.stat().st_size for p in allfiles)
fields=defaultdict(lambda:dict(copies=0,bytes=0,unique={}))
full={};trialbytes=0;schema=Counter();sumdomain=0;dupwithin=defaultdict(lambda:[0,0,set()]);formatter_match=True
for p in paths:
 data=p.read_bytes();d=json.loads(data);trialbytes+=len(data);meta=d['result']['domain_randomization'];schema[tuple(meta.keys())]+=1
 formatter_match &= data== encode(d).encode() or data== (encode(d)+'\n').encode()
 k=hashlib.sha256(json.dumps(meta,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest();n=prop_size('domain_randomization',meta,4);sumdomain+=n
 info=full.setdefault(k,dict(copies=0,bytes=0,one_property_bytes=n,canonical_payload_bytes=len(encode(meta).encode())));info['copies']+=1;info['bytes']+=n
 group=str(p.relative_to(RUN)).split('/evaluations/')[0];dupwithin[group][0]+=1;dupwithin[group][1]+=n;dupwithin[group][2].add(k)
 for key in ('robot_xml','assets','asset_sha256'):
  v=meta[key];n=prop_size(key,v,6);valuehash=hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest();a=fields[key];a['copies']+=1;a['bytes']+=n;a['unique'].setdefault(valuehash,n)
metadata_unique_bytes=sum(v['one_property_bytes'] for v in full.values());duplicate_bytes=sumdomain-metadata_unique_bytes
# Conservative new shared-object store: full payload once per hash, ~100 bytes
# of JSON lookup overhead per store entry, ~110-byte reference in every trial.
refs=len(paths)*110;store=sum(v['canonical_payload_bytes']+100 for v in full.values());net=sumdomain-refs-store
out=dict(checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),scope='Frozen path listing of committed trial.json during active run; no source/result mutation',trial_files=len(paths),trial_bytes=trialbytes,all_run_bytes=totalbytes,formatter_matches_existing_files=formatter_match,domain_metadata_bytes=sumdomain,unique_domain_objects=len(full),one_copy_metadata_property_bytes=metadata_unique_bytes,identical_domain_repetition_bytes=duplicate_bytes,duplicate_share_of_trials=duplicate_bytes/trialbytes,duplicate_share_of_entire_run=duplicate_bytes/totalbytes,conservative_reference_store_net_saving_bytes=net,reference_store_assumption='110 bytes per trial reference + one full metadata JSON payload and100 bytes lookup overhead per unique hash; estimate, not a migrated artifact',fields={k:dict(copies=v['copies'],bytes=v['bytes'],unique_values=len(v['unique']),one_copy_bytes=sum(v['unique'].values()),identical_repetition_bytes=v['bytes']-sum(v['unique'].values())) for k,v in fields.items()},domain_changed_within_same_evaluation_folder=sum(len(v[2])>1 for v in dupwithin.values()),other_fields_not_assumed_redundant=['per-frame solver residuals/status','per-frame clearance','candidate bounds','kinematic metrics'],snapshot_paths_sha256=hashlib.sha256('\n'.join(sorted(str(p) for p in paths)).encode()).hexdigest())
(OUT/'storage_redundancy.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
