"""Export read-only offline annotation status for the complete dataset viewer."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def snapshot(run):
    run = Path(run).resolve()
    selected = json.loads((run / 'selection.json').read_text())['sources']
    episodes = {}
    newest = (run / 'selection.json').stat().st_mtime
    for source in selected:
        folder = run / source['suite'] / f"task_{source['task_id']:02d}" / source['demo']
        status = 'running' if (folder / 'prepared_input.json').exists() else 'pending'
        evidence = None
        for name in ('annotation_status.json', 'adapter_error.json'):
            path = folder / name
            if path.exists():
                evidence = json.loads(path.read_text())
                status = evidence['status']
                newest = max(newest, path.stat().st_mtime)
                break
        key = f"{source['suite']}/{source['task_id']:02d}/{source['demo']}"
        episodes[key] = dict(status=status, source_episode_index=source['episode_index'],
                             evidence=str(folder), error=evidence.get('error') if evidence else None)
    return dict(updated_at=datetime.fromtimestamp(newest, timezone.utc).isoformat(),
                selected=len(selected), counts=dict(Counter(e['status'] for e in episodes.values())),
                source_run=str(run), episodes=episodes)


def export(run, viewer):
    result = snapshot(run)
    destination = Path(viewer) / 'annotation_status.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(destination)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--viewer', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.viewer)['counts'], ensure_ascii=False))
