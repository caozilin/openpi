import json
from pathlib import Path

from examples.libero.export_dataset_annotation_status import snapshot


def test_status_keeps_every_selected_demo_and_original_ids(tmp_path):
    sources = [dict(suite='libero_spatial', task_id=0, demo=f'demo_{n}', episode_index=i)
               for i, n in enumerate((0, 2, 17, 49))]
    (tmp_path / 'selection.json').write_text(json.dumps({'sources': sources}))
    for number, filename, status in [(0, 'annotation_status.json', 'labeled'),
                                      (2, 'annotation_status.json', 'no_feasible_label'),
                                      (17, 'adapter_error.json', 'adapter_error')]:
        folder = tmp_path / 'libero_spatial/task_00' / f'demo_{number}'
        folder.mkdir(parents=True)
        (folder / filename).write_text(json.dumps({'status': status}))
    result = snapshot(tmp_path)
    assert result['selected'] == 4
    assert result['counts'] == dict(labeled=1, no_feasible_label=1, adapter_error=1, pending=1)
    assert set(result['episodes']) == {f'libero_spatial/00/demo_{n}' for n in (0, 2, 17, 49)}
    assert result['episodes']['libero_spatial/00/demo_17']['status'] == 'adapter_error'
