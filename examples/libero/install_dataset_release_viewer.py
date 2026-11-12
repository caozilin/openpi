"""Install recorded-release UI assets without rebuilding videos or traces."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'data/libero_tasktol_joint_servo/20261025_success10_v1/viewer'


def install(directory):
    directory = Path(directory)
    index = directory / 'index.html'
    html = index.read_text()
    if 'assets/tolerance_comparison.js' not in html:
        raise ValueError('Dataset viewer has no recorded tolerance comparison')
    files = {
        'release_values.js': ROOT / 'examples/libero/benchmark_viewer/release_values.js',
        **{name: ROOT / 'examples/libero/mask_viewer' / name for name in
           ('dataset_release_values.js', 'dataset_release.js', 'dataset_release.css')},
    }
    for name, source in files.items():
        shutil.copy2(source, directory / 'assets' / name)
    if 'assets/dataset_release.css' not in html:
        html = html.replace('</head>', '<link rel="stylesheet" href="assets/dataset_release.css">\n</head>')
    if 'assets/dataset_release.js' not in html:
        scripts = ''.join(f'<script src="assets/{name}"></script>' for name in
                          ('release_values.js', 'dataset_release_values.js', 'dataset_release.js'))
        html = html.replace('</body>', scripts + '\n</body>')
    index.write_text(html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    install(args.directory)
    print(f'Recorded-release UI installed: {args.directory}')
