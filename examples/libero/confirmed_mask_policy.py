"""Causal task-mask selection for the live LIBERO controller.

Finger phases use the existing shared classifier. Semantic identities use only
current LIBERO state, never baseline frame numbers or future policy actions.
Repeated grasps keep the object's mask. Native predicates only route masks;
they do not alter the policy's observation, actions, or success criterion.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEGACY_CONFIRMED = ROOT / 'data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1/confirmed_masks.json'
CONFIRMED = ROOT / 'examples/libero/configs/libero40_semantic_tolerances_deg.json'
PHASES = {'pregrasp': 'APP', 'grasp': 'ENG', 'postgrasp': 'EXEC', 'release': 'DIS'}
# The two mechanisms have different masks from the preceding transported item.
MECHANISM_SWITCH = {
    3: ('in', 'akita_black_bowl_1', 'white_cabinet_1_bottom_region'),
    9: ('in', 'white_yellow_mug_1', 'microwave_1_heating_region'),
}
# Long06/07 can have different angles for their two objects; proximity is queried only
# while approaching, then the identity is latched through grasp/carry/release.
DISTINCT_OBJECTS = ('alphabet_soup_1', 'cream_cheese_1')
OBJECT_IDENTITIES = {6: ('porcelain_mug_1', 'chocolate_pudding_1'), 7: DISTINCT_OBJECTS}


def load_masks(path=CONFIRMED):
    table = json.loads(Path(path).read_text())
    assert table['mask_order'] == ['xF', 'yF', 'zF']
    numeric = table.get('schema') == 'libero40_semantic_tolerances_deg.1'
    if numeric:
        assert table['angle_unit'] == 'degree' and table['bound_semantics'] == 'symmetric_half_width'
    if not numeric: assert table['bounds_deg'] == [30] * 6
    table['tolerance_bounds_policy'] = 'per_task_subtask_phase' if numeric else 'uniform'
    assert len(table['tasks']) == 40
    keys = [(t['suite'], t['task_id']) for t in table['tasks']]
    assert set(keys) == {(s,t) for s in ('libero_spatial','libero_object','libero_goal','libero_10') for t in range(10)}
    assert sum(len(t['subtasks']) for t in table['tasks']) == 49
    for task in table['tasks']:
        assert [s['index'] for s in task['subtasks']] == list(range(len(task['subtasks'])))
        for subtask in task['subtasks']:
            if numeric:
                degrees = {}
                for phase in ('APP', 'ENG', 'EXEC', 'DIS'):
                    vector = subtask[phase]
                    assert isinstance(vector,list) and len(vector)==3
                    assert all(type(v) in (int,float) and np.isfinite(v) and 0<=v<=180 for v in vector)
                    degrees[phase] = list(vector)
                    subtask[phase] = ''.join('1' if v>0 else '0' for v in vector)
                assert degrees['ENG'] == degrees['DIS'] == [0,0,0]
                subtask['phase_tolerances_deg'] = degrees
                subtask['phase_bounds_deg'] = {p: np.repeat(v,2).tolist() for p,v in degrees.items()}
            else:
                subtask['phase_tolerances_deg'] = {p:[30 if bit=='1' else 0 for bit in subtask[p]] for p in ('APP','ENG','EXEC','DIS')}
                subtask['phase_bounds_deg'] = {p:list(table['bounds_deg']) for p in ('APP','ENG','EXEC','DIS')}
            assert subtask['ENG'] == subtask['DIS'] == '000'
            for phase in ('APP', 'ENG', 'EXEC', 'DIS'):
                assert len(subtask[phase]) == 3 and set(subtask[phase]) <= {'0', '1'}
    return table


def table_metadata(table):
    if table['tolerance_bounds_policy']=='per_task_subtask_phase':
        return dict(semantic_mask_policy='user_edited_task_phase_angles',
                    tolerance_bounds_policy='per_task_subtask_phase', tolerance_bounds_deg=None)
    return dict(semantic_mask_policy='user_confirmed_task_phase_masks',
                tolerance_bounds_policy='uniform', tolerance_bounds_deg=list(table['bounds_deg']))


def freeze_default_config(run):
    """Create an immutable per-run copy and the request fields that select it."""
    table = load_masks()
    destination = Path(run).resolve() / 'confirmed_masks.json'
    contents = CONFIRMED.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != contents:
            raise ValueError('Existing run has different frozen tolerances; use a new run directory')
    else:
        with destination.open('xb') as stream:
            stream.write(contents)
    return dict(table_metadata(table), confirmed_masks_path=str(destination),
                confirmed_masks_sha256=hashlib.sha256(contents).hexdigest(),
                source_workbook_sha256=table['source_workbook_sha256'])


def step_with_selection(session, action, selection):
    """Pass the authored directional angles to the actual control session."""
    observation, row = session.step(action, phase=selection['phase'],
                                   subtask_index=selection['subtask_index'],
                                   semantic_mask=selection['semantic_mask'],
                                   bounds_rad=selection['bounds_rad'])
    assert row['semantic_mask'] == selection['semantic_mask']
    assert np.allclose(row['bounds_rad'], selection['bounds_rad'], atol=1e-15, rtol=0)
    row.update({k:v for k,v in selection.items() if k not in ('phase','subtask_index','semantic_mask','bounds_rad')})
    return observation, row


def object_route_names(task):
    if task['suite']!='libero_10' or task['task_id'] not in OBJECT_IDENTITIES:
        return None
    signatures = {tuple(s['phase_tolerances_deg']['APP']+s['phase_tolerances_deg']['EXEC']) for s in task['subtasks']}
    return OBJECT_IDENTITIES[task['task_id']] if len(signatures)>1 else None


class ConfirmedMaskPolicy:
    def __init__(self, suite, task_id, *, table=None, classifier=None):
        table = load_masks() if table is None else table
        self.task = next(t for t in table['tasks'] if (t['suite'], t['task_id']) == (suite, task_id))
        self.object_route_names = object_route_names(self.task)
        if classifier is None:
            # Import from the caller's frozen runtime before the legacy adapter
            # adds the shared project's root to sys.path for model metadata.
            from tasktol.tasks.phases import PhaseClassifier
            from examples.libero.mask_viewer.phase_adapter import shared_classifier
            cls, model, _ = shared_classifier()
            assert cls is PhaseClassifier
            classifier = cls(model)
        self.classifier = classifier
        self.semantic = 0
        self.reference_index = 0
        self.previous_key = None
        self.bowl_grasped = False

    def resolve(self, fingers, action, domain, eef_position):
        observed = self.classifier.update(np.clip(np.asarray(fingers) * [1., -1.], 0., .04), bool(action[6] > 0))
        phase = PHASES[observed.phase.key]
        suite, tid = self.task['suite'], self.task['task_id']
        details = {}
        if suite == 'libero_10' and tid in MECHANISM_SWITCH:
            predicate = MECHANISM_SWITCH[tid]
            done = bool(domain._eval_predicate(list(predicate)))
            details['semantic_boundary_predicate'] = list(predicate)
            details['semantic_boundary_satisfied'] = done
            # Carry/release retains the transported object's identity. Switching
            # at the next approach avoids treating placement as a new mechanism.
            if phase == 'APP' and done:
                self.semantic = 1
        elif self.object_route_names is not None:
            distances = [float(np.linalg.norm(np.asarray(domain.sim.data.body_xpos[domain.obj_body_id[name]]) - eef_position)) for name in self.object_route_names]
            details['semantic_target_distances_m'] = distances
            details['semantic_target_objects'] = list(self.object_route_names)
            if phase == 'APP' or self.previous_key is None:
                self.semantic = int(np.argmin(distances))
        else:
            # These object subtasks have the same complete angle settings;
            # mechanisms use the explicit placement predicate above.
            self.semantic = min(observed.subtask_index, len(self.task['subtasks']) - 1)
        if suite == 'libero_goal' and tid == 3:
            obj = domain.objects_dict['akita_black_bowl_1']
            contact = bool(domain._check_grasp(domain.robots[0].gripper, obj.contact_geoms))
            if phase == 'APP':
                self.bowl_grasped = False
            self.bowl_grasped = self.bowl_grasped or contact
            # Confirmed exception: opening the drawer through bowl approach is
            # APP. ENG/DIS remain strict, including during drawer manipulation.
            if phase == 'EXEC' and not self.bowl_grasped:
                phase = 'APP'
            details['semantic_bowl_grasped'] = self.bowl_grasped
        key = (observed.subtask_index, self.semantic)
        if self.previous_key is not None and key != self.previous_key:
            self.reference_index += 1
        self.previous_key = key
        subtask = self.task['subtasks'][self.semantic]
        mask = subtask[phase]
        bounds = subtask['phase_bounds_deg'][phase]
        return dict(phase=phase, subtask_index=self.reference_index, semantic_mask=mask,
                    semantic_tolerance_deg=list(subtask['phase_tolerances_deg'][phase]),
                    semantic_bounds_deg=list(bounds), bounds_rad=np.deg2rad(bounds).tolist(),
                    semantic_subtask_index=self.semantic,
                    semantic_subtask_label=self.task['subtasks'][self.semantic]['label'],
                    observed_phase=observed.phase.key, observed_subtask_index=observed.subtask_index,
                    phase_aperture_m=observed.aperture_m, phase_motion_m=observed.motion_m,
                    semantic_assignment_source='current native state and shared live finger phase; no future/baseline frames',
                    **details)
