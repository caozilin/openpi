import json
import pytest
from examples.libero.export_training_viewer_selection import snapshot


def fixtures(tmp_path):
    identity=dict(suite='libero_spatial',task_id=0,demo='demo_2')
    request=dict(tolerance_episodes=1,tasks=[dict(suite='libero_spatial',task_id=0,selected_demos=['demo_2'])],
                 entries=[dict(episode_index=2,source=identity,label=dict(sha256='label'),
                               replay=dict(success=True,execution_rejected=False,folder=str(tmp_path/'recording')))])
    catalog=dict(datasets=[dict(suite='libero_spatial',task_id=0,episodes=[dict(path='demo_2',tolerance=dict(folder=str(tmp_path/'recording'),video='exact.mp4'))])])
    return request,catalog


def test_exact_membership_and_source_reference(tmp_path):
    request,catalog=fixtures(tmp_path);p=tmp_path/'request.json';p.write_text(json.dumps(request))
    out=snapshot(p,catalog)
    assert out['selected']==1
    assert out['episodes']=={'libero_spatial/00/demo_2':dict(episode_index=2,video='exact.mp4',label_sha256='label')}


@pytest.mark.parametrize('error',['unsuccessful','wrong_source','unselected','duplicate'])
def test_do_not_infer_selection_from_labels_or_other_rollouts(tmp_path,error):
    request,catalog=fixtures(tmp_path)
    if error=='unsuccessful':request['entries'][0]['replay']['success']=False
    if error=='wrong_source':catalog['datasets'][0]['episodes'][0]['tolerance']['folder']=str(tmp_path/'other')
    if error=='unselected':request['tasks'][0]['selected_demos']=['demo_3']
    if error=='duplicate':request['entries']*=2
    p=tmp_path/'request.json';p.write_text(json.dumps(request))
    with pytest.raises(AssertionError):snapshot(p,catalog)
