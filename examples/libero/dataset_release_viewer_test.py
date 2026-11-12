import importlib.util
from pathlib import Path


def test_installer_is_idempotent_and_preserves_recorded_data(tmp_path):
    source = Path(__file__).with_name('install_dataset_release_viewer.py')
    spec = importlib.util.spec_from_file_location('release_installer', source)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'index.html').write_text('<head></head><body><script src="assets/tolerance_comparison.js"></script></body>')
    trace = tmp_path / 'trajectory.json'
    trace.write_bytes(b'[{"rotation_release_rad":[0,0.1,0]}]')
    original = trace.read_bytes()
    installer.install(tmp_path)
    installer.install(tmp_path)
    html = (tmp_path / 'index.html').read_text()
    assert html.count('src="assets/dataset_release.js"') == 1
    assert html.count('href="assets/dataset_release.css"') == 1
    assert html.index('tolerance_comparison.js') < html.index('dataset_release.js')
    assert trace.read_bytes() == original
    assert all((tmp_path / 'assets' / name).is_file() for name in
               ['release_values.js', 'dataset_release_values.js', 'dataset_release.js', 'dataset_release.css'])
