"""Install the nominal/tolerance × main/wrist layout, without touching source data."""
from pathlib import Path
import re


ASSETS = Path(__file__).parent / 'mask_viewer'


def install(viewer):
    viewer = Path(viewer)
    document = (viewer / 'index.html').read_text()
    document = re.sub(r'<label><input id="compareSource".*?</label>', '', document)
    document = re.sub(r'<figure id="sourceFigure">.*?</figure>', '', document, flags=re.S)
    document = re.sub(r'<a id="openSourceVideo".*?</a>', '', document)
    document = re.sub(r'<h2>[^<]*· 主视角 · 20 Hz</h2>', '<h2>无容差 / 标注容差 · 主视角与腕部 · 20 Hz</h2>', document)
    document = document.replace('video-grid rollout-comparison triple-comparison', 'video-grid rollout-comparison quad-comparison')
    document = document.replace('video-grid rollout-comparison"', 'video-grid rollout-comparison quad-comparison"')
    if 'id="nominalWristVideo"' not in document:
        marker = '<div id="episodeScope"'
        end = document.index(marker)
        grid_end = document.rfind('</div>', 0, end)
        wrists = '''<figure id="nominalWristFigure"><video id="nominalWristVideo" preload="metadata" muted playsinline></video><figcaption>无容差重构 · 腕部视角 · 20 Hz</figcaption></figure>
<figure id="toleranceWristFigure"><video id="toleranceWristVideo" preload="metadata" muted playsinline></video><figcaption id="toleranceWristCaption">标注容差 · 腕部视角 · 20 Hz</figcaption></figure>
'''
        document = document[:grid_end] + wrists + document[grid_end:]
    if 'assets/video_sync.js' not in document:
        document = document.replace('<script src="assets/tolerance_comparison.js">', '<script src="assets/video_sync.js"></script><script src="assets/tolerance_comparison.js">')
    # Put video before the lengthy mask summary so all four views are visible on arrival.
    content = '<div id="content" class="content hidden">'
    if content in document:
        content_start = document.index(content) + len(content)
        video_start = document.index('<section class="panel video-panel">', content_start)
        if document[content_start:video_start].strip():
            video_end = document.index('</section>', video_start) + len('</section>')
            video = document[video_start:video_end]
            document = document[:video_start] + document[video_end:]
            document = document[:content_start] + '\n' + video + document[content_start:]
    for name in ['rollout_adapter.js', 'tolerance_comparison.js', 'annotated_tolerance_comparison.js',
                 'full_dataset.js', 'rollout_comparison.css', 'video_sync.js']:
        text = (ASSETS / name).read_text()
        if name == 'rollout_adapter.js':
            text = text.removesuffix('loadDatasets();\n')
        (viewer / 'assets' / name).write_text(text)
    (viewer / 'index.html').write_text(document)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('viewer', type=Path)
    install(parser.parse_args().viewer)
