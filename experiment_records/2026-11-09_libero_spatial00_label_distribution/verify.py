"""Recompute reported marginal statistics from exported per-demo CSV."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE/'summary.json').read_text())
sources = json.loads((HERE/'sources.json').read_text())
details = json.loads((HERE/'details.json').read_text())
for source in sources['files']:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
with (HERE/'per_demo.csv').open() as f:
    rows = list(csv.DictReader(f))
assert len(rows)==47 and len({r['demo'] for r in rows})==47
for unit, record in summary['directions'].items():
    for field, suffix in [('search','cap_deg'),('actual','peak_deg')]:
        x = [float(r[unit+' '+suffix]) for r in rows]
        result = record[field]
        assert math.isclose(st.fmean(x),result['mean'],abs_tol=1e-10)
        assert math.isclose(st.pstdev(x),result['std'],abs_tol=1e-10)
        assert math.isclose(st.median(x),result['median'],abs_tol=1e-10)
        pairs=[abs(a-b) for i,a in enumerate(x) for b in x[i+1:]]
        assert len(pairs)==1081
        assert math.isclose(st.fmean(pairs),result['pairwise_abs_mean'],abs_tol=1e-10)
        assert sum(abs(v)<=1e-6 for v in x)==result['zero_count']
        assert min(x)==result['min'] and max(x)==result['max']
        if field=='search':
            maximum=30 if unit.startswith('APP') else 15
            assert sum(abs(v-maximum)<=1e-6 for v in x)==result['upper_count']
assert all(sum(d['phase_counts'].values())==d['frames'] for d in details['demos'])
result=dict(complete=True,source_files_checked=len(sources['files']),demos_checked=47,
            signed_directions_checked=8,pairs_per_direction=1081,
            original_raw_data_modified=False,new_search_or_simulation=False,
            scope='SHA binding, phase labels and independent CSV summary recomputation; historical execution audit reused')
(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
