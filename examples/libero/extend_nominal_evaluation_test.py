from examples.libero.extend_nominal_evaluation import assignments


def test_six_replica_partition_preserves_every_new_task_seed_exactly_once():
    groups = assignments()
    triples = [(x['suite'], x['task_id'], x['seed']) for group in groups.values() for x in group]
    assert len(triples) == len(set(triples)) == 160
    assert set(z for _, _, z in triples) == {6, 7, 8, 9}
    assert sorted(map(len, groups.values())) == [26, 26, 27, 27, 27, 27]
    assert all(len({x['seed'] for x in group}) == 2 for group in groups.values())
    for suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
        for task in range(10):
            assert {z for s, t, z in triples if s == suite and t == task} == {6, 7, 8, 9}
