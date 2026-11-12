"""Export exact frozen tolerance replacement membership, without copying media."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUEST = ROOT / 'data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109/replacement_request.json'
DEFAULT_VIEWER = ROOT / 'data/libero_tasktol_joint_servo/20261101_full_v1/viewer'


def snapshot(request_path, catalog):
    raw = Path(request_path).read_bytes()
    request = json.loads(raw)
    available = {(d['suite'], d['task_id'], e['path']): e for d in catalog['datasets'] for e in d['episodes']}
    declared = {(t['suite'], t['task_id'], demo) for t in request['tasks'] for demo in t['selected_demos']}
    episodes = {}
    for entry in request['entries']:
        source, replay = entry['source'], entry['replay']
        identity = (source['suite'], source['task_id'], source['demo'])
        key = f"{identity[0]}/{identity[1]:02d}/{identity[2]}"
        assert key not in episodes and identity in available and identity in declared
        assert replay['success'] and not replay['execution_rejected']
        shown = available[identity]['tolerance']
        assert Path(shown['folder']).resolve() == Path(replay['folder']).resolve()
        episodes[key] = dict(episode_index=entry['episode_index'], video=shown['video'],
                             label_sha256=entry['label']['sha256'])
    assert len(episodes) == request['tolerance_episodes'] == len(declared)
    return dict(schema='libero_training_tolerance_selection/1',
                dataset='libero_reconstructed_joint13_successmix20_20261109',
                selected=len(episodes), rule='exact membership in frozen replacement_request.entries; no resampling',
                request=str(Path(request_path).resolve()), request_sha256=hashlib.sha256(raw).hexdigest(),
                episodes=episodes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', type=Path, default=DEFAULT_REQUEST)
    ap.add_argument('--viewer', type=Path, default=DEFAULT_VIEWER)
    args = ap.parse_args()
    result = snapshot(args.request, json.loads((args.viewer / 'viewer_catalog.json').read_text()))
    (args.viewer / 'training_selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f"Published {result['selected']} exact training selections; no media copied")


if __name__ == '__main__':
    main()
