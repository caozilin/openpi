"""Read-only validation of the published four-camera catalog; starts no server."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.libero.export_training_viewer_selection import snapshot
from examples.libero.rollout_quad_viewer_test import Tags


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    viewer = ROOT / 'data/libero_tasktol_joint_servo/20261101_full_v1/viewer'
    catalog = json.loads((viewer / 'viewer_catalog.json').read_text())
    exported = json.loads((viewer / 'wrist_export_verification.json').read_text())
    assert exported['complete'] and exported['episodes'] == 1875 and exported['tolerance_episodes'] == 1851
    episodes = [e for d in catalog['datasets'] for e in d['episodes']]
    assert len(episodes) == len({e['key'] for e in episodes}) == 1875
    for episode in episodes:
        wrist = viewer / episode['wrist_video_path']
        assert wrist.is_file()
        saved = json.loads(wrist.with_suffix('.json').read_text())
        assert saved['frames'] == episode['frames'] and saved['frequency_hz'] == 20
        assert saved['display_transform'] == 'rotate180'
        assert Path(saved['source']) == Path(episode['selected_root']) / 'recording.hdf5'
        assert saved['video_sha256'] == sha(wrist)
        if episode.get('tolerance'):
            t = episode['tolerance']
            assert os.path.samefile(viewer / t['wrist_video'], Path(t['folder']) / 'wrist_rgb.mp4')
    parser = Tags()
    html = (viewer / 'index.html').read_text()
    parser.feed(html)
    assert parser.videos == ['mainVideo', 'toleranceVideo', 'nominalWristVideo', 'toleranceWristVideo']
    assert not {'sourceVideo', 'sourceFigure', 'openSourceVideo', 'compareSource'} & set(parser.ids)
    assert len(parser.ids) == len(set(parser.ids))
    for name in ['rollout_adapter.js', 'tolerance_comparison.js', 'annotated_tolerance_comparison.js',
                 'full_dataset.js', 'video_sync.js', 'rollout_comparison.css']:
        published = (viewer / 'assets' / name).read_text()
        source = (ROOT / 'examples/libero/mask_viewer' / name).read_text()
        assert published == (source.removesuffix('loadDatasets();\n') if name == 'rollout_adapter.js' else source)
    request = ROOT / 'data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109/replacement_request.json'
    assert sha(request) == '92e86f597f84b801299ba80482adf506ea1a989e8b94f7a4cec53619797d8337'
    current = snapshot(request, catalog)
    selected = json.loads((viewer / 'training_selection.json').read_text())
    assert current['episodes'] == selected['episodes'] and current['selected'] == selected['selected'] == 301
    result = dict(complete=True, layout=[['nominal_main', 'tolerance_main'], ['nominal_wrist', 'tolerance_wrist']],
                  nominal_wrist=1875, tolerance_wrist=1851, training_selected=301,
                  all_nominal_wrist_sha_verified=True, tolerance_reuses_original_inodes=True,
                  all_wrist_fps_resolution_and_frame_counts_verified_during_export=True,
                  missing_tolerance=24, no_source_file_deletion=True,
                  no_server_or_browser_started=True, source_export_receipt=str(viewer / 'wrist_export_verification.json'),
                  viewer=str(viewer), request_sha256=sha(request))
    output = Path(__file__).with_name('quad_verification.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
