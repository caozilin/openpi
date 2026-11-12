from types import SimpleNamespace as NS

import numpy as np
import pytest

from examples.libero.confirmed_mask_policy import ConfirmedMaskPolicy, LEGACY_CONFIRMED, load_masks, step_with_selection
from examples.libero.eval_confirmed_mask_libero_local import assignments


class Classifier:
    def __init__(self):
        self.phase = 'pregrasp'
        self.cycle = 0

    def update(self, fingers, closed):
        return NS(phase=NS(key=self.phase), subtask_index=self.cycle, aperture_m=float(sum(fingers)), motion_m=0.)


class Domain:
    done = False
    grasp = False
    objects_dict = {'akita_black_bowl_1': NS(contact_geoms=['bowl'])}
    robots = [NS(gripper='gripper')]
    obj_body_id = {'alphabet_soup_1': 0, 'cream_cheese_1': 1, 'porcelain_mug_1': 0, 'chocolate_pudding_1': 1}
    sim = NS(data=NS(body_xpos=np.array([[0., 0, 0], [1., 0, 0]])))

    def _eval_predicate(self, predicate):
        return self.done

    def _check_grasp(self, gripper, geoms):
        return self.grasp


def resolve(policy, domain=None, position=(0., 0, 0)):
    return policy.resolve([.04, -.04], [0.] * 6 + [-1.], domain or Domain(), np.asarray(position))


def test_all_400_combinations_assigned_once_to_six_replicas():
    jobs = [j for group in assignments().values() for j in group]
    assert len(jobs) == len({(j['suite'], j['task_id'], j['seed']) for j in jobs}) == 400
    assert sorted(map(len, assignments().values())) == [66, 66, 67, 67, 67, 67]
    assert {j['seed'] for j in jobs} == set(range(10))


@pytest.mark.parametrize('task', load_masks()['tasks'])
def test_every_table_phase_used_and_regrasp_keeps_confirmed_masks(task):
    # Multi-identity special handling has dedicated tests below.
    if task['suite'] == 'libero_goal' and task['task_id'] == 3:
        return
    classifier = Classifier()
    p = ConfirmedMaskPolicy(task['suite'], task['task_id'], classifier=classifier)
    for name, phase in [('pregrasp', 'APP'), ('grasp', 'ENG'), ('postgrasp', 'EXEC'), ('release', 'DIS')]:
        classifier.phase = name
        r = resolve(p)
        assert r['semantic_mask'] == task['subtasks'][r['semantic_subtask_index']][phase]
    classifier.phase, classifier.cycle = 'pregrasp', 8
    r = resolve(p)
    assert r['semantic_mask'] == task['subtasks'][r['semantic_subtask_index']]['APP']
    assert r['subtask_index'] == 1


@pytest.mark.parametrize('tid', [3, 9])
def test_mechanism_switch_requires_actual_placement_and_new_approach(tid):
    c, d = Classifier(), Domain()
    p = ConfirmedMaskPolicy('libero_10', tid, classifier=c)
    assert resolve(p, d)['semantic_mask'] == '010'
    c.phase, c.cycle = 'pregrasp', 1  # A regrasp alone must not select mechanism.
    assert resolve(p, d)['semantic_subtask_index'] == 0
    c.phase, d.done = 'postgrasp', True
    assert resolve(p, d)['semantic_subtask_index'] == 0
    c.phase = 'pregrasp'
    r = resolve(p, d)
    assert r['semantic_subtask_index'] == 1 and r['semantic_mask'] == '000'


def test_long07_distinguishes_objects_and_latches_identity_while_closed():
    c = Classifier()
    p = ConfirmedMaskPolicy('libero_10', 7, classifier=c)
    assert resolve(p, position=(.9, 0, 0))['semantic_mask'] == '010'
    c.phase = 'postgrasp'
    assert resolve(p, position=(0, 0, 0))['semantic_subtask_index'] == 1
    c.phase, c.cycle = 'pregrasp', 1
    assert resolve(p, position=(0, 0, 0))['semantic_mask'] == '011'


def test_goal03_drawer_is_approach_until_bowl_grasp_but_eng_dis_strict():
    c, d = Classifier(), Domain()
    p = ConfirmedMaskPolicy('libero_goal', 3, classifier=c)
    c.phase = 'postgrasp'
    assert resolve(p, d)['semantic_mask'] == '010'
    c.phase = 'grasp'
    assert resolve(p, d)['semantic_mask'] == '000'
    c.phase, d.grasp = 'postgrasp', True
    r = resolve(p, d)
    assert r['semantic_mask'] == '111' and r['semantic_tolerance_deg'] == [5,5,5]
    c.phase = 'release'
    assert resolve(p, d)['semantic_mask'] == '000'


def test_long02_is_not_uniform111():
    c = Classifier()
    p = ConfirmedMaskPolicy('libero_10', 2, classifier=c)
    for cycle in [0, 1, 2, 8]:
        c.cycle, c.phase = cycle, 'pregrasp'
        assert resolve(p)['semantic_mask'] == '010'
        c.phase = 'postgrasp'
        assert resolve(p)['semantic_mask'] == '000'


def test_long06_routes_distinct_carry_angles_and_latches_object_during_carry():
    c = Classifier()
    p = ConfirmedMaskPolicy('libero_10', 6, classifier=c)
    assert resolve(p)['semantic_subtask_index'] == 0
    c.phase = 'postgrasp'
    white = resolve(p, position=(.99,0,0))
    assert white['semantic_tolerance_deg'] == [5,5,15]
    assert white['semantic_subtask_index'] == 0
    c.phase, c.cycle = 'pregrasp', 1
    assert resolve(p, position=(.99,0,0))['semantic_subtask_index'] == 1
    c.phase = 'postgrasp'
    pudding = resolve(p, position=(0,0,0))
    assert pudding['semantic_tolerance_deg'] == [15,15,15]
    assert pudding['semantic_subtask_index'] == 1
    # A repeated grasp cycle near pudding does not revert to white-mug angles.
    c.phase, c.cycle = 'pregrasp', 2
    assert resolve(p, position=(.99,0,0))['semantic_subtask_index'] == 1
    c.phase = 'postgrasp'
    assert resolve(p)['semantic_tolerance_deg'] == [15,15,15]


def test_explicit_legacy_table_preserves_original_binary_masks_and_uniform_bounds():
    table = load_masks(LEGACY_CONFIRMED)
    c, d = Classifier(), Domain()
    p = ConfirmedMaskPolicy('libero_goal', 3, table=table, classifier=c)
    c.phase, d.grasp = 'postgrasp', True
    assert resolve(p,d)['semantic_mask'] == '000'
    for task in table['tasks']:
        for sub in task['subtasks']:
            assert all(v == [30]*6 for v in sub['phase_bounds_deg'].values())
    # The old Long06 angles are identical; preserve its historical routing.
    p = ConfirmedMaskPolicy('libero_10', 6, table=table, classifier=c)
    assert p.object_route_names is None


def test_control_session_receives_degrees_converted_to_actual_directional_radians():
    c = Classifier()
    p = ConfirmedMaskPolicy('libero_spatial',4,classifier=c)
    selection = resolve(p)
    class ControlSession:
        def step(self, action, **settings):
            self.settings = settings
            return 'observation', dict(settings)
    session = ControlSession()
    obs, row = step_with_selection(session, [0]*7, selection)
    assert obs == 'observation'
    np.testing.assert_allclose(session.settings['bounds_rad'],np.deg2rad([0,0,5,5,0,0]),atol=1e-15,rtol=0)
    assert session.settings['semantic_mask'] == '010'
    assert row['semantic_tolerance_deg'] == [0,5,0]
    c.phase = 'grasp'
    _, row = step_with_selection(session,[0]*7,resolve(p))
    assert session.settings['bounds_rad'] == [0]*6 and row['semantic_mask'] == '000'


def test_control_session_cannot_silently_substitute_fixed_thirty_degree_bounds():
    class WrongSession:
        def step(self, action, **settings):
            return None, dict(settings,bounds_rad=[np.pi/6]*6)
    selection = resolve(ConfirmedMaskPolicy('libero_spatial',4,classifier=Classifier()))
    with pytest.raises(AssertionError):
        step_with_selection(WrongSession(),[0]*7,selection)
