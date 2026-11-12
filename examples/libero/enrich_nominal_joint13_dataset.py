"""Append audited signed tolerance labels without touching original PNG/actions/state."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openpi.shared.libero_joint_dataset import SOURCE_SCHEMA, TOLERANCE_ORDER, validate_info, validate_tolerance

def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def label_values(source, label, table):
    n=source['frames'];origin=label['source']
    for key in ('episode_index','suite','task_id','demo','frames','recording_sha256'):
        if source[key]!=origin[key]:raise ValueError(f'Label source mismatch: {key}')
    if label['units']!='rad' or label['order']!=list(TOLERANCE_ORDER) or label['control_hz']!=20:
        raise ValueError('Wrong tolerance units/order/frequency')
    frames=label['frames']
    if len(frames)!=n or table.num_rows!=n:raise ValueError('Label/frame count mismatch')
    if [f['source_frame_index'] for f in frames]!=list(range(n)):raise ValueError('Labels are not in source frame order')
    np.testing.assert_array_equal(table['frame_index'].to_numpy(),np.arange(n))
    np.testing.assert_array_equal(table['episode_index'].to_numpy(),np.full(n,source['episode_index']))
    np.testing.assert_allclose(table['timestamp'].to_numpy(),[f['time_s'] for f in frames],atol=2e-6,rtol=0)
    values=validate_tolerance([f['rotation_tolerance_rad'] for f in frames],leading_shape=(n,))
    for f,v in zip(frames,values,strict=True):
        if f['phase'] not in ('pregrasp','postgrasp') and np.any(v):raise ValueError('Transition-phase labels must remain zero')
    return values

def append_columns(table,source,label):
    if 'rotation_tolerance' in table.column_names or 'trajectory_is_tolerance' in table.column_names:
        raise ValueError('Refuse to append onto an already enriched table')
    values=label_values(source,label,table)
    result=table.append_column('rotation_tolerance',pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(),type=pa.float32()),type=pa.list_(pa.field('element',pa.float32()),6)))
    result=result.append_column('trajectory_is_tolerance',pa.array(np.zeros(table.num_rows,dtype=np.int64)))
    metadata=dict(result.schema.metadata or {});hf=json.loads(metadata[b'huggingface'])
    hf['info']['features'].update(rotation_tolerance={'feature':{'dtype':'float32','_type':'Value'},'length':6,'_type':'Sequence'},trajectory_is_tolerance={'dtype':'int64','_type':'Value'})
    metadata[b'huggingface']=json.dumps(hf).encode();result=result.replace_schema_metadata(metadata)
    for col in table.column_names:
        if not result[col].equals(table[col]):raise AssertionError(f'Original column changed: {col}')
    return result,values

def feature_stats(values):
    return {**{k:f(values,axis=0).tolist() for k,f in [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]},'count':[len(values)]}

def collect_labels(sources, label_roots):
    entries=[];audits=[read(p/'annotation_independent_final_audit.json') for p in label_roots]
    audit_shas={str(p):sha(p/'annotation_independent_final_audit.json') for p in label_roots}
    if not all(a['complete'] for a in audits):raise ValueError('Require complete independent annotation audits')
    for s in sources:
        key=(s['suite'],s['task_id'],s['demo']);matches=[]
        for root,a in zip(label_roots,audits,strict=True):
            row=next((r for r in a['results'] if (r['suite'],r['task_id'],r['demo'])==key and r['status']=='labeled'),None)
            if row is not None:
                p=root/s['suite']/f"task_{s['task_id']:02d}"/s['demo']/'tolerance_labels.json'
                assert sha(p)==row['labels_sha256'];matches.append(dict(path=str(p),sha256=sha(p),audit=str(root/'annotation_independent_final_audit.json'),audit_sha256=audit_shas[str(root)]))
        if len(matches)!=1:raise ValueError(f'Need exactly one audited label for {key}; found {len(matches)}')
        entries.append(dict(source=s,label=matches[0],trajectory_is_tolerance=0))
    return entries

def export(source_root,output,label_roots):
    source_root=source_root.resolve();output=output.resolve()
    if output==source_root or output.is_relative_to(source_root):raise ValueError('Output must be independent of original dataset')
    original_info=read(source_root/'meta/info.json');plan=read(source_root/'meta/reconstruction_sources.json')
    assert original_info['total_episodes']==1875 and original_info['total_frames']==312374 and original_info['fps']==20
    entries=collect_labels(plan['sources'],label_roots)
    request=dict(source_root=str(source_root),source_index_sha256=sha(source_root/'meta/reconstruction_sources.json'),source_info_sha256=sha(source_root/'meta/info.json'),schema=SOURCE_SCHEMA,episodes=entries,png_policy='identical embedded bytes; no decode/re-encode',all_trajectories='nominal')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'conversion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        req=output/'conversion_request.json'
        if req.exists() and read(req)!=request:raise ValueError('Frozen dataset request changed')
        write(req,request)
        remaining=sum((source_root/original_info['data_path'].format(episode_chunk=e['source']['episode_index']//1000,episode_index=e['source']['episode_index'])).stat().st_size for e in entries if not (output/'receipts'/f"{e['source']['episode_index']:06d}.json").exists())
        if shutil.disk_usage(output).free<remaining+(2<<30):raise RuntimeError('Insufficient free space for independent enriched Parquet plus2GiB reserve')
        receipts=[];stats=[json.loads(x) for x in (source_root/'meta/episodes_stats.jsonl').read_text().splitlines()]
        for entry in entries:
            s=entry['source'];i=s['episode_index'];relative=original_info['data_path'].format(episode_chunk=i//1000,episode_index=i)
            src=source_root/relative;dest=output/relative;receipt_path=output/'receipts'/f'{i:06d}.json'
            label=read(Path(entry['label']['path']));original=pq.read_table(src);enriched,values=append_columns(original,s,label)
            source_sha=sha(src)
            if receipt_path.exists():
                receipt=read(receipt_path)
                assert receipt['source_sha256']==source_sha and receipt['label_sha256']==entry['label']['sha256'] and receipt['output_sha256']==sha(dest)
                restored=pq.read_table(dest)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.parquet');pq.write_table(enriched,tmp,compression='snappy');restored=pq.read_table(tmp)
                assert restored.equals(enriched,check_metadata=True);tmp.replace(dest)
                receipt=dict(episode_index=i,frames=s['frames'],source_sha256=source_sha,label_sha256=entry['label']['sha256'],output_path=relative,output_sha256=sha(dest),all_original_columns_identical=True,trajectory_is_tolerance=0)
                write(receipt_path,receipt)
            assert restored.equals(enriched,check_metadata=True)
            st=copy.deepcopy(stats[i]);assert st['episode_index']==i
            st['stats']['rotation_tolerance']=feature_stats(values);st['stats']['trajectory_is_tolerance']=feature_stats(np.zeros((len(values),1),np.int64));stats[i]=st
            receipts.append(receipt)
            write(output/'conversion_state.json',dict(stage='converting',processed=i+1,total=1875))
            if (i+1)%25==0:print(f'Converted {i+1}/1875',flush=True)
        info=copy.deepcopy(original_info);info['source_schema']=SOURCE_SCHEMA
        info['features'].update(rotation_tolerance=dict(dtype='float32',shape=[6],names=list(TOLERANCE_ORDER)),trajectory_is_tolerance=dict(dtype='int64',shape=[1],names=['trajectory_is_tolerance']))
        info['joint13_contract']=dict(rotation_tolerance_units='rad',rotation_tolerance_order=list(TOLERANCE_ORDER),trajectory_types={'0':'nominal','1':'tolerance'},trajectory_counts={'nominal':1875,'tolerance':0},physical_action_units='meters/world_rotation_vector_radians/gripper; no OSC scaling or extra delta',effective_training_dimensions=13,model_interface_dimensions=32)
        validate_info(info);(output/'meta').mkdir(exist_ok=True)
        for name in ('episodes.jsonl','tasks.jsonl','reconstruction_sources.json'):
            shutil.copyfile(source_root/'meta'/name,output/'meta'/name)
        shutil.copyfile(source_root/'meta/reconstruction_audit.json',output/'meta/source_reconstruction_audit.json')
        write(output/'meta/info.json',info)
        (output/'meta/episodes_stats.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in stats))
        write(output/'meta/tolerance_sources.json',dict(schema=SOURCE_SCHEMA,episodes=entries))
        write(output/'conversion_manifest.json',dict(episodes=receipts))
        write(output/'conversion_completion.json',dict(complete=True,episodes=1875,frames=312374,request_sha256=sha(req),all_nominal=True,original_png_and_base_columns_identical=True,independent_audit_pending=True))
        write(output/'conversion_state.json',dict(stage='complete',processed=1875,total=1875))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--labels',type=Path,action='append',required=True);a=p.parse_args();export(a.source,a.output,a.labels)
if __name__=='__main__':main()
