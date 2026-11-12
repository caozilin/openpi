"""Compare exported joint19 run metadata against the completed state/action7 run."""
import copy
import csv
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'data/wandb_exports/openpi_2026-10-26/runs/franka_mujoco_v2_joint19_lora__ucflosn1'
CURRENT = ROOT / 'analysis/pi05_state_action7_15k_20261028'
old = json.loads((OLD / 'metadata.json').read_text())
current = json.loads((CURRENT / 'resolved_config.json').read_text())
plan = json.loads((CURRENT / 'training_command.json').read_text())
# resolved_config is the saved config object; apply actual command-line path/name overrides.
current = copy.deepcopy(current)
args = plan['argv'][4:]
for index in range(0, len(args), 2):
    key, value = args[index:index + 2]
    assert key.startswith('--')
    parts = key[2:].replace('-', '_').split('.')
    node = current
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
current['checkpoint_dir_actual'] = plan['checkpoint_dir']
with (OLD / 'history.csv').open(encoding='utf-8-sig') as stream:
    old_history = list(csv.DictReader(stream))
log = (ROOT / 'analysis/pi05_bs24_vs_bs84_20261029/baseline_bs84_train.log').read_text(errors='replace')
current_history = []
for step, line in re.findall(r'Step\s+(\d+): ([^\n]+)', log):
    values = {k: float(v) for k, v in re.findall(r'([\w/]+)=([-+0-9.eE]+)', line)}
    current_history.append({'step': int(step), **values})

def flatten(value, prefix=''):
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        result.update(flatten(child, f'{prefix}.{key}' if prefix else key))
    return result

left, right = flatten(old['config']), flatten(current)
changed, same, old_only, current_only = {}, {}, {}, {}
for key in sorted(left.keys() | right.keys()):
    if key not in right:
        old_only[key] = left[key]
    elif key not in left:
        current_only[key] = right[key]
    elif left[key] != right[key]:
        changed[key] = {'joint19': left[key], 'state_action7': right[key]}
    else:
        same[key] = left[key]
windows = {}
for lo, hi in [(14500, 15000), (29600, 30100)]:
    rows = [r for r in old_history if lo <= int(r['_step']) <= hi and r['loss/action']]
    windows[f'old_{lo}_{hi}'] = {'points': len(rows), **{k: statistics.mean(float(r[k]) for r in rows)
                                                        for k in ['loss/action', 'loss/stage_target_pose', 'loss/rotation_tolerance', 'loss']}}
rows = [r for r in current_history if 14500 <= r['step'] <= 15000]
windows['current_14500_15000'] = {'points': len(rows), 'loss/action': statistics.mean(r['loss/action'] for r in rows)}
result = {'old_run_name': old['run_name'], 'old_run_id': old['run_id'], 'old_actual_last_step': int(old['summary']['_step']),
          'current_run_name': plan['experiment'], 'current_actual_last_record': current_history[-1],
          'changed_shared_fields': changed, 'unchanged_shared_fields': same,
          'old_only_fields': old_only, 'current_only_fields': current_only,
          'loss_windows': windows,
          'training_samples_at_30k_old': 30000 * 24, 'training_samples_actual_old': 30100 * 24,
          'training_samples_current': 15000 * 84,
          'evidence_limits': ['Old exact dataset snapshot, norm statistics and code SHA are unavailable.',
                              'Old resume=True; restoration checkpoint source is not established by metadata.',
                              'Training losses use different data, targets and norm statistics; not common-set evaluation.',
                              'Current comparison means the completed 7GPU batch84 15k run; bs24 short runs are separate.']}
(OUT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
(OUT / 'old_metadata.json').write_text(json.dumps(old, indent=2) + '\n')
(OUT / 'current_actual_config.json').write_text(json.dumps(current, indent=2) + '\n')
print(json.dumps({'changed': changed, 'loss_windows': windows}, indent=2))
