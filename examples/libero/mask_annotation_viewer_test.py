"""LIBERO alignment, mask provenance, and local video-serving regressions."""
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

builder=module('libero_mask_builder','build_mask_annotation_viewer.py')
server=module('libero_mask_server','serve_mask_annotation_viewer.py')

def test_selection_keeps_representative_and_samples_only_other_successes():
    task=dict(suite='libero_10',task_id=2,demo_id='demo_12',task_name='example')
    meta={'episodes':{f'demo_{i}':{'success':i!=7} for i in range(25)}}
    selected=builder.choose_episodes(task,meta,20261019)
    assert selected[0]=='demo_12'
    assert len(set(selected))==10
    assert 'demo_7' not in selected
    assert selected==builder.choose_episodes(task,meta,20261019)


def test_closure_axis_calibration_and_degeneracy_do_not_swap_axes():
    frame,h=builder.observed_gravity_frame([0,0,0,1])
    assert np.allclose(frame,np.eye(3)) and h==1
    frame,h=builder.observed_gravity_frame([np.sqrt(.5),0,0,np.sqrt(.5)])
    assert frame is None and h<1e-6
    # Rotating the tool around world Z must rotate the horizontal closure heading.
    frame,h=builder.observed_gravity_frame([0,0,np.sqrt(.5),np.sqrt(.5)])
    assert np.allclose(np.asarray(frame)[:,1],[-1,0,0],atol=1e-8)


def test_stability_requires_point_two_seconds_and_observed_aperture_response():
    commands=np.r_[-np.ones(4),np.ones(14)]
    width=np.r_[np.full(5,.08),.06,.04,np.full(11,.03)]
    events,candidates=builder.gripper_events(commands,width)
    assert events[0]['command_frame']==4
    assert events[0]['first_aperture_response']>=5
    assert events[0]['stable_frame']-events[0]['first_aperture_response']>=4
    assert candidates[4]=='gripper_closing'
    events,_=builder.gripper_events(commands,np.full(18,.08))
    assert events[0]['stable_frame'] is None


def test_observation_alignment_and_extra_demo_masks_remain_unassigned(tmp_path):
    n=12;source=np.arange(n)+3;source[7:]+=4
    h5=tmp_path/'data.hdf5'
    with h5py.File(h5,'w') as f:
        for demo in ['demo_4','demo_5']:
            g=f.create_group('data/'+demo)
            actions=np.zeros((n,7));actions[:,6]=-1;actions[4:,6]=1;g['actions']=actions
            ee=np.zeros((n,6));ee[:,0]=np.arange(n)*.001;ee[:,2]=.9;g['obs/ee_states']=ee
            joints=np.tile(np.arange(n)[:,None]*.01,(1,7));g['obs/joint_states']=joints
            fingers=np.tile([.03,-.03],(n,1));g['obs/gripper_states']=fingers
            robot=np.c_[fingers,ee[:,:3],np.tile([0,0,0,1],(n,1))];g['tasktol/pre_action_robot_states']=robot
            g['tasktol/source_step_indices']=source
            g['obs/agentview_rgb']=np.zeros((n,256,256,3),dtype=np.uint8)
            # Deliberately incompatible legacy data must not be consumed.
            g['states']=np.full((n,5),99.0)
    source_run=tmp_path/'frozen';(source_run/'videos').mkdir(parents=True);(source_run/'videos/main.mp4').write_bytes(b'original-video')
    output=tmp_path/'viewer';output.mkdir()
    task=dict(suite='libero_object',task_id=1,task_name='example',demo_id='demo_4',language='put object in basket',video_path='videos/main.mp4',subtasks=[dict(index=0,phase_spans=[dict(phase='APP',processed_interval=[0,4]),dict(phase='ENG',processed_interval=[4,7]),dict(phase='EXEC',processed_interval=[7,n])],APP={'effective_mask':'010'},EXEC={'effective_mask':'111'})])
    before=copy.deepcopy(task);sel={'processed_hdf5':str(h5),'source_hdf5':str(h5)};metadata={'retained_steps':n,'source_step_indices':source.tolist()}
    rec=builder.export_episode((task,sel,metadata,'demo_4',0,source_run,output));data=json.loads((output/rec['trace_path']).read_text())
    assert data['trajectory'][7]['time_s']==.35
    assert data['trajectory'][7]['source_frame']==14
    assert data['trajectory'][7]['joint_state']['velocity_rad_s']==[.2]*7
    assert data['trajectory'][0]['joint_state']['velocity_rad_s']==[None]*7
    assert data['trajectory'][0]['nominal_target_pose'] is None
    assert data['trajectory'][7]['execution_mask']=='111'
    assert data['trajectory'][0]['ee_state']['position_m'][2]==.9
    assert data['trajectory'][0]['gripper_state']['aperture_m']==.06
    extra_video=output/'videos/libero_object/example/demo_5.mp4';extra_video.write_bytes(b'extra-video')
    extra=builder.export_episode((task,sel,metadata,'demo_5',1,source_run,output));other=json.loads((output/extra['trace_path']).read_text())
    assert all(f['subtask_index'] is None and f['execution_mask'] is None for f in other['trajectory'])
    assert all(f['phase_source']=='gripper_event_candidate_only' for f in other['trajectory'])
    assert task==before


@pytest.mark.parametrize(('header','expected'), [('bytes=0-3',(0,3)),('bytes=5-',(5,31)),('bytes=-4',(28,31)),('bytes=0-99',(0,31)),(None,None)])
def test_ranges(header,expected):assert server.byte_range(header,32)==expected

@pytest.mark.parametrize('header',['bytes=32-','bytes=8-3','bytes=0-2,4-5','bytes=-0','invalid'])
def test_invalid_ranges(header):
    with pytest.raises(ValueError):server.byte_range(header,32)


def test_http_handler_seeks_video_and_rejects_path_escape(tmp_path):
    (tmp_path/'index.html').write_text('<h1>viewer</h1>');(tmp_path/'clip.mp4').write_bytes(bytes(range(32)))
    class Request:
        def __init__(self,path,range_header=''):
            self.raw=io.BytesIO(f'GET {path} HTTP/1.0\r\nHost: localhost\r\n{range_header}\r\n'.encode());self.response=bytearray()
        def makefile(self,*args,**kwargs):return self.raw
        def sendall(self,data):self.response.extend(data)
    class Handler(server.ViewerHandler):
        def log_message(self,*args):pass
    request=Request('/clip.mp4','Range: bytes=8-15\r\n')
    Handler(request,('127.0.0.1',1234),SimpleNamespace(server_name='localhost',server_port=8768),directory=str(tmp_path))
    head,body=bytes(request.response).split(b'\r\n\r\n',1)
    assert b'206 Partial Content' in head and b'Content-Range: bytes 8-15/32' in head
    assert body==bytes(range(8,16))
    target=tmp_path.parent/'private.txt';target.write_text('private')
    (tmp_path/'escape').symlink_to(target)
    request=Request('/escape');Handler(request,('127.0.0.1',1234),SimpleNamespace(server_name='localhost',server_port=8768),directory=str(tmp_path))
    assert b'403 Outside viewer directory' in bytes(request.response)
    assert b'private' not in bytes(request.response)
