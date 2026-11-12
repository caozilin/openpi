import copy,json
import numpy as np
import pyarrow as pa
import pytest
from examples.libero.enrich_nominal_joint13_dataset import append_columns
from openpi.shared.libero_joint_dataset import TOLERANCE_ORDER

def fixture():
    source=dict(episode_index=9,suite='libero_spatial',task_id=0,demo='demo_9',frames=2,recording_sha256='recording')
    label=dict(source=source.copy(),units='rad',order=list(TOLERANCE_ORDER),control_hz=20,frames=[dict(source_frame_index=i,time_s=i/20,phase='pregrasp' if i==0 else 'grasp',rotation_tolerance_rad=[.1]*6 if i==0 else [0.]*6) for i in range(2)])
    table=pa.table({'image':[b'original PNG0',b'original PNG1'],'wrist_image':[b'wrist0',b'wrist1'],'state':pa.array([[float(x)]*8 for x in range(2)],type=pa.list_(pa.field('element',pa.float32()),8)),'actions':pa.array([[float(x)]*7 for x in range(2)],type=pa.list_(pa.field('element',pa.float32()),7)),'frame_index':[0,1],'episode_index':[9,9],'timestamp':pa.array([0,.05],type=pa.float32())}).replace_schema_metadata({b'huggingface':json.dumps({'info':{'features':{}}}).encode()})
    return source,label,table

def test_export_preserves_every_original_column_and_nominal_flag():
    s,l,t=fixture();out,v=append_columns(t,s,l)
    for name in t.column_names:assert out[name].equals(t[name])
    assert out['trajectory_is_tolerance'].to_pylist()==[0,0]
    np.testing.assert_array_equal(v[1],np.zeros(6));assert out['rotation_tolerance'].type==pa.list_(pa.float32(),6)

@pytest.mark.parametrize('kind',['frame_order','source','transition','radians'])
def test_bad_labels_cannot_silently_enter_dataset(kind):
    s,l,t=fixture();l=copy.deepcopy(l)
    if kind=='frame_order':l['frames'][0]['source_frame_index']=1
    if kind=='source':l['source']['demo']='demo_8'
    if kind=='transition':l['frames'][1]['rotation_tolerance_rad']=[.1]*6
    if kind=='radians':l['units']='degrees'
    with pytest.raises((ValueError,AssertionError)):append_columns(t,s,l)


def test_parquet_roundtrip_preserves_values_and_huggingface_metadata(tmp_path):
    import pyarrow.parquet as pq
    s,l,t=fixture();out,_=append_columns(t,s,l);p=tmp_path/'episode.parquet';pq.write_table(out,p,compression='snappy')
    assert out.equals(pq.read_table(p),check_metadata=True)
