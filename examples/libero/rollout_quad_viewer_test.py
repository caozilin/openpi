from html.parser import HTMLParser
from pathlib import Path

from examples.libero.install_rollout_quad_viewer import install


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'video':
            self.videos.append(attrs['id'])


def test_layout_removes_source_adds_two_wrist_views_and_is_idempotent(tmp_path):
    viewer = tmp_path / 'viewer'
    viewer.mkdir()
    (viewer / 'assets').mkdir()
    (viewer / 'index.html').write_text('''<html><head></head><body>
<h2>原始 / 无容差 / 标注容差 · 主视角 · 20 Hz</h2>
<div class="comparison-toolbar"><label><input id="compareSource" checked>原始</label></div>
<div class="video-grid rollout-comparison triple-comparison">
<figure><video id="mainVideo"></video></figure><figure id="sourceFigure"><video id="sourceVideo"></video></figure>
<figure id="toleranceFigure"><video id="toleranceVideo"></video></figure></div>
<div id="episodeScope"></div><a id="openSourceVideo">原始视频</a>
<script src="assets/tolerance_comparison.js"></script></body></html>''')
    install(viewer)
    first = (viewer / 'index.html').read_text()
    install(viewer)
    assert (viewer / 'index.html').read_text() == first
    parser = Tags()
    parser.feed(first)
    assert parser.videos == ['mainVideo', 'toleranceVideo', 'nominalWristVideo', 'toleranceWristVideo']
    assert len(parser.ids) == len(set(parser.ids))
    assert not set(parser.ids) & {'sourceFigure', 'sourceVideo', 'compareSource', 'openSourceVideo'}
    assert 'quad-comparison' in first and 'triple-comparison' not in first
    assert first.index('assets/video_sync.js') < first.index('assets/tolerance_comparison.js')
    assert not (viewer / 'assets/rollout_adapter.js').read_text().endswith('loadDatasets();\n')


def test_export_rotates_saved_wrist_once_and_retains_source(tmp_path):
    import h5py
    import imageio.v2 as imageio
    import numpy as np
    from examples.libero.prepare_rollout_wrist_videos import digest, export_wrist

    source = tmp_path / 'recording.hdf5'
    images = np.zeros((3, 256, 256, 3), dtype=np.uint8)
    images[:, :128] = [255, 0, 0]
    images[:, 128:] = [0, 0, 255]
    with h5py.File(source, 'w') as recording:
        recording.create_dataset('data/demo_7/obs/eye_in_hand_rgb', data=images)
    before = digest(source)
    target = tmp_path / 'viewer/wrist_rgb.mp4'
    saved = export_wrist(tmp_path, 'demo_7', 3, target)
    assert saved['frames'] == 3 and saved['display_transform'] == 'rotate180'
    with imageio.get_reader(target) as reader:
        frames = [frame for frame in reader]
    assert len(frames) == 3
    assert np.abs(frames[0].astype(float) - images[0, ::-1, ::-1]).mean() < 3
    assert digest(source) == before
    assert export_wrist(tmp_path, 'demo_7', 3, target) == saved
