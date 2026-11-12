import hashlib
import json

import numpy as np
import pytest

from examples.libero.confirmed_mask_policy import CONFIRMED, LEGACY_CONFIRMED, freeze_default_config, load_masks
from examples.libero.eval_confirmed_mask_libero_local import validate_tolerance_request
from examples.libero.import_semantic_tolerances import WORKBOOK, convert, read_cells, SUITES


def workbook_config():
    return convert(read_cells(WORKBOOK), json.loads(LEGACY_CONFIRMED.read_text()))


def test_all_294_user_angles_match_frozen_runtime_vectors_exactly_without_changing_workbook():
    before = hashlib.sha256(WORKBOOK.read_bytes()).hexdigest()
    converted = workbook_config()
    actual = json.loads(CONFIRMED.read_text())
    assert actual['source_workbook_sha256'] == before
    assert actual == dict(converted, source_workbook_sha256=before)
    raw_cells = read_cells(WORKBOOK)
    indexed = {(t['suite'],t['task_id'],s['index']):s for t in load_masks()['tasks'] for s in t['subtasks']}
    assert len(indexed) == 49
    count = 0
    for row in range(9,58):
        sub = indexed[SUITES[raw_cells[f'A{row}'][0]],int(raw_cells[f'B{row}'][0]),int(raw_cells[f'K{row}'][0])]
        for phase,columns in [('APP','EFG'),('EXEC','HIJ')]:
            expected = [float(raw_cells[f'{column}{row}'][0]) for column in columns]
            assert sub['phase_tolerances_deg'][phase] == expected
            assert sub['phase_bounds_deg'][phase] == np.repeat(expected,2).tolist()
            assert sub[phase] == ''.join('1' if a>0 else '0' for a in expected)
            count += 3
        for phase in ('ENG','DIS'):
            assert sub[phase] == '000' and sub['phase_bounds_deg'][phase] == [0]*6
    assert count == 294
    assert hashlib.sha256(WORKBOOK.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('cell,value',[
    ('E9',('-1',None,False)), ('E9',('181',None,False)),
    ('E9',('nan',None,False)), ('E9',(None,None,False)),
    ('E9',('5','s',False)), ('E9',('5',None,True)),
    ('E8',('EXEC_xF(°)','s',False)), ('A9',('Unknown','s',False)),
    ('K9',('0.5',None,False)), ('B10',('0',None,False)),
])
def test_import_rejects_invalid_angles_headers_identifiers_or_duplicate_rows(cell,value):
    cells = read_cells(WORKBOOK)
    cells[cell] = value
    with pytest.raises(ValueError):
        convert(cells,json.loads(LEGACY_CONFIRMED.read_text()))


@pytest.mark.parametrize('value',[[-1,30,0],[181,30,0],[float('nan'),30,0],[True,30,0],['5',30,0],[5,30]])
def test_runtime_also_rejects_invalid_hand_edited_config(tmp_path,value):
    table = workbook_config()
    table['tasks'][0]['subtasks'][0]['APP'] = value
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(table))
    with pytest.raises(AssertionError):
        load_masks(path)


def test_runtime_rejects_nonzero_transition_stage_angles(tmp_path):
    table = workbook_config()
    table['tasks'][0]['subtasks'][0]['ENG'] = [0,5,0]
    path = tmp_path/'bad.json';path.write_text(json.dumps(table))
    with pytest.raises(AssertionError):load_masks(path)


def test_new_run_freezes_default_angles_and_rejects_wrong_request_or_existing_config(tmp_path):
    run = tmp_path/'new_run'
    request = freeze_default_config(run)
    assert (run/'confirmed_masks.json').read_bytes() == CONFIRMED.read_bytes()
    assert request['tolerance_bounds_policy'] == 'per_task_subtask_phase'
    assert request['tolerance_bounds_deg'] is None
    assert validate_tolerance_request(request,run)['source_workbook_sha256'] == request['source_workbook_sha256']
    assert freeze_default_config(run) == request
    wrong = dict(request,tolerance_bounds_deg=[30]*6)
    with pytest.raises(AssertionError):validate_tolerance_request(wrong,run)
    with pytest.raises(AssertionError):validate_tolerance_request(dict(request,confirmed_masks_path=str(CONFIRMED)),run)
    contents = json.loads((run/'confirmed_masks.json').read_text())
    contents['tasks'][0]['subtasks'][0]['APP'][1] = 15
    (run/'confirmed_masks.json').write_text(json.dumps(contents))
    with pytest.raises(AssertionError):validate_tolerance_request(request,run)
    with pytest.raises(ValueError,match='new run'):freeze_default_config(run)


def test_old_uniform_angle_request_still_validates_only_its_explicit_legacy_file():
    request = dict(confirmed_masks_path=str(LEGACY_CONFIRMED),
                   confirmed_masks_sha256=hashlib.sha256(LEGACY_CONFIRMED.read_bytes()).hexdigest(),
                   semantic_mask_policy='user_confirmed_task_phase_masks',tolerance_bounds_deg=[30]*6)
    assert validate_tolerance_request(request)['tolerance_bounds_policy'] == 'uniform'
