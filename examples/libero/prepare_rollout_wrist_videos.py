"""Export saved nominal wrist observations; link existing tolerance wrist MP4s.

This only prepares viewer media. It never steps a simulator or changes recordings.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import imageio.v2 as imageio
import numpy as np


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def probe(path):
    result = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,r_frame_rate,width,height', '-of', 'json', str(path),
    ]))['streams'][0]
    assert result['r_frame_rate'] == '20/1', (path, result)
    assert result['width'] == result['height'] == 256, (path, result)
    return int(result['nb_read_frames'])


def export_wrist(original, demo, frames, target):
    """Reuse only a validated export bound to this saved image dataset."""
    original, target = Path(original), Path(target)
    source = original / 'recording.hdf5'
    dataset = f'data/{demo}/obs/eye_in_hand_rgb'
    metadata = target.with_suffix('.json')
    with h5py.File(source, 'r') as recording:
        images = recording[dataset][()]
    assert images.shape == (frames, 256, 256, 3) and images.dtype == np.uint8
    pixels_sha = hashlib.sha256(images.tobytes()).hexdigest()
    if metadata.exists() and target.exists():
        saved = json.loads(metadata.read_text())
        assert saved['source_images_sha256'] == pixels_sha
        assert saved['video_sha256'] == digest(target)
        assert saved['frames'] == frames
        return saved
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + '.partial.mp4')
    with imageio.get_writer(temporary, fps=20, codec='libx264', quality=8,
                           macro_block_size=1, output_params=['-threads', '1']) as writer:
        for frame in images:
            writer.append_data(np.ascontiguousarray(frame[::-1, ::-1]))
    assert probe(temporary) == frames
    temporary.replace(target)
    saved = dict(source=str(source), image_dataset=dataset, source_images_sha256=pixels_sha,
                 video_sha256=digest(target), frames=frames, frequency_hz=20,
                 display_transform='rotate180', width=256, height=256,
                 provenance='saved reconstructed pre-action wrist pixels; no simulation')
    save_json(metadata, saved)
    return saved


def prepare_episode(item):
    viewer, episode = item
    viewer = Path(viewer)
    relative = Path(episode['video_path']).with_name('wrist_rgb.mp4')
    saved = export_wrist(episode['selected_root'], episode['path'], episode['frames'], viewer / relative)
    result = dict(key=episode['key'], wrist_video_path=str(relative), frames=saved['frames'],
                  video_sha256=saved['video_sha256'])
    tolerance = episode.get('tolerance')
    if tolerance:
        source = Path(tolerance['folder']) / 'wrist_rgb.mp4'
        target_relative = Path(tolerance['video']).with_name('wrist_rgb.mp4')
        target = viewer / target_relative
        assert source.is_file(), source
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert os.path.samefile(source, target) or digest(source) == digest(target)
        else:
            os.link(source, target)
        assert probe(target) == tolerance['frames']
        result.update(tolerance_wrist_video=str(target_relative), tolerance_video_sha256=digest(target))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    viewer = args.viewer.resolve()
    # One publisher/exporter per viewer; a stale lock is advisory evidence, not auto-deleted.
    import fcntl
    with (viewer / 'wrist_export.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalog = json.loads((viewer / 'viewer_catalog.json').read_text())
        episodes = [e for d in catalog['datasets'] for e in d['episodes']]
        started = time.time()
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(prepare_episode, [(str(viewer), e) for e in episodes]):
                results.append(result)
                save_json(viewer / 'wrist_export_state.json', dict(
                    stage='exporting', completed=len(results), total=len(episodes), pid=os.getpid(),
                    elapsed_seconds=round(time.time() - started, 1)))
        by_key = {r['key']: r for r in results}
        for episode in episodes:
            result = by_key[episode['key']]
            episode['wrist_video_path'] = result['wrist_video_path']
            if episode.get('tolerance'):
                episode['tolerance']['wrist_video'] = result['tolerance_wrist_video']
        save_json(viewer / 'viewer_catalog.json', catalog)
        script = viewer / 'assets/catalog.js'
        temp = script.with_suffix('.tmp')
        temp.write_text('window.LIBERO_CATALOG=' + json.dumps(catalog, ensure_ascii=False, separators=(',', ':')) + ';\n')
        temp.replace(script)
        save_json(viewer / 'wrist_export_verification.json', dict(
            complete=True, episodes=len(results), tolerance_episodes=sum('tolerance_wrist_video' in r for r in results),
            entries=results, elapsed_seconds=round(time.time() - started, 1), no_simulation=True))
        save_json(viewer / 'wrist_export_state.json', dict(stage='complete', completed=len(results), total=len(episodes)))
        print(json.dumps(dict(complete=True, episodes=len(results), elapsed_seconds=time.time() - started)))


if __name__ == '__main__':
    main()
