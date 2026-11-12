"""Verify the audit-only adapter keeps source IDs while bounding future queries."""
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import ast
from pathlib import Path
import pytest

def reader_class(path):
    tree=ast.parse(Path(path).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='SelectedEpisodeReader')
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));scope={'LeRobotDataset':LeRobotDataset}
    exec(compile(module,str(path),'exec'),scope)
    return scope['SelectedEpisodeReader']

@pytest.mark.parametrize('script',['examples/libero/audit_nominal_joint13_dataset.py','examples/libero/verify_joint13_upload_server.py'])
def test_noncontiguous_episode_ids_map_to_compact_offsets_and_pad_at_own_end(script):
    ds=object.__new__(reader_class(script))
    ds.episodes=[0,47,1400]
    ds.episode_data_index={'from':torch.tensor([0,3,8]),'to':torch.tensor([3,8,10])}
    ds.delta_indices={'actions':list(range(10)),'rotation_tolerance':list(range(10))}
    query,pad=ds._get_query_indices(7,47)
    assert query['actions']==[7]*10 and query['rotation_tolerance']==[7]*10
    assert pad['actions_is_pad'].tolist()==[False]+[True]*9
    query,pad=ds._get_query_indices(8,1400)
    assert query['actions']==[8]+[9]*9
    assert pad['rotation_tolerance_is_pad'].tolist()==[False,False]+[True]*8
    assert ds.episodes==[0,47,1400]
