from mask_viewer.phase_adapter import classify_rows


def rows(widths, commands):
    return [dict(action=[0]*6+[1 if c else -1],pre_action=dict(finger_qpos_m=[w/2,-w/2])) for w,c in zip(widths,commands,strict=True)]


def test_20hz_three_stable_frames_latching_and_subtask_cycle():
    widths=[.08]*5+[.03]*8+[.01,.04,.02,.05]+[.08]*9
    commands=[False]*5+[True]*12+[False]*9
    result,_,rule=classify_rows(rows(widths,commands))
    assert rule['frequency_hz']==20 and rule['stable_frames']==3
    assert result[5]['phase']=='grasp'
    assert all(r['phase_sample_frame']==i and r['phase_sampled'] for i,r in enumerate(result))
    assert result[7]['phase']=='grasp'
    assert result[8]['phase']=='postgrasp'
    assert all(r['phase']=='postgrasp' for r in result[8:17])  # Motion doesn't unlatch.
    assert result[17]['phase']=='release'
    assert result[20]['phase']=='pregrasp'
    assert result[20]['observed_subtask_index']==1


def test_future_commands_cannot_rewrite_prefix():
    source=rows([.08]*5+[.03]*15,[False]*5+[True]*15)
    prefix,_,_=classify_rows(source[:13]);full,_,_=classify_rows(source)
    assert prefix==full[:13]
