"""Read existing Spatial00 labels and selected trials; never run a simulation."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107')
OLD = Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_dsp16_first5pertask_20261106')
REPO = Path('/media/czl/sata/franka_my_code/openpi')
PLAN = REPO/'data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/plan.json'
H5 = REPO/'third_party/libero/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5'
UNITS = ['APP Ry-', 'APP Ry+', 'EXEC Rx-', 'EXEC Rx+', 'EXEC Ry-', 'EXEC Ry+', 'EXEC Rz-', 'EXEC Rz+']
INDICES = [2, 3, 6, 7, 8, 9, 10, 11]
UPPER = np.array([30, 30, 15, 15, 15, 15, 15, 15])
evidence = []


def read(path, expected=None):
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected is not None:
        assert digest == expected, path
    evidence.append(dict(path=str(path), sha256=digest))
    return json.loads(content)


def stats(x, upper=None):
    x = np.asarray(x, dtype=float)
    pairs = np.abs(x[:, None]-x[None, :])[np.triu_indices(len(x), 1)]
    out = dict(n=len(x), mean=float(x.mean()), std=float(x.std()), min=float(x.min()),
               q25=float(np.quantile(x, .25)), median=float(np.median(x)), q75=float(np.quantile(x, .75)),
               max=float(x.max()), span=float(np.ptp(x)), pairwise_abs_mean=float(pairs.mean()),
               zero_count=int(np.isclose(x, 0, atol=1e-6, rtol=0).sum()))
    if upper is not None:
        out['upper_count'] = int(np.isclose(x, upper, atol=1e-6, rtol=0).sum())
        out['interior_count'] = len(x)-out['zero_count']-out['upper_count']
    return out


def load_demo(folder, expected=None):
    labels = read(folder/'tolerance_labels.json', expected)
    prepared = read(folder/'prepared_input.json')
    # This task has exactly one observed cycle in every selected demonstration.
    assert len(labels['subtasks']) == len(prepared['subtasks']) == 1
    annotation = read(folder/'subtask_00/annotation.json')
    best = annotation['best']
    caps = np.array(best['actual_pre_caps_deg']+best['actual_post_caps_deg'])
    assert np.allclose(caps, labels['subtasks'][0]['pregrasp']+labels['subtasks'][0]['postgrasp'])
    assert np.all(caps[[0, 1, 4, 5]] == 0)
    assert prepared['subtasks'][0]['APP_caps_deg'] == [0., 30., 0.]
    assert prepared['subtasks'][0]['EXEC_caps_deg'] == [15., 15., 15.]
    phase_counts = Counter()
    for frame in labels['frames']:
        phase_counts[frame['phase']] += 1
        values = np.rad2deg(frame['rotation_tolerance_rad'])
        if frame['phase'] == 'pregrasp':
            assert np.allclose(values, caps[:6])
        elif frame['phase'] == 'postgrasp':
            assert np.allclose(values, caps[6:])
        else:
            assert np.allclose(values, 0)
    raw = read(best['raw_trial_path'], best['raw_trial_sha256'])
    ev = raw['result']['evaluation']
    assert ev['trajectory_complete'] and ev['kinematic_feasible_without_table']
    ext = ev['sqp_subtask_stage_release_extrema_deg'][0]
    peak = []
    for phase in ['pregrasp', 'postgrasp']:
        peak.extend([ext[phase][side][axis] for axis in range(3) for side in ['negative', 'positive']])
    full = next(o for o in annotation['observations'] if np.allclose(
        o['actual_pre_caps_deg']+o['actual_post_caps_deg'], annotation['design']['maximum_caps_deg']))
    return dict(demo=labels['source']['demo'], episode_index=labels['source']['episode_index'],
                caps_deg=caps[INDICES].tolist(), actual_peak_deg=np.array(peak)[INDICES].tolist(),
                frames=len(labels['frames']), phase_counts=dict(phase_counts),
                best_velocity=best['velocity_rad_s'], full_velocity=full['velocity_rad_s'],
                full_pass=full['success'], selected_query=best['query_index'],
                full_caps=bool(np.allclose(caps, annotation['design']['maximum_caps_deg'])),
                source_folder=prepared['source']['folder'])


def main():
    plan = read(PLAN)
    request = read(ROOT/'annotation_request.json')
    selection = read(ROOT/'selection.json')
    audit = read(ROOT/'annotation_independent_final_audit.json', plan['audit_sha256'])
    assert audit['complete'] and audit['per_frame_labels_verified']
    entries = sorted([e for e in plan['entries'] if e['task']['suite']=='libero_spatial'
                      and e['task']['task_id']==0], key=lambda e:int(e['job']['demo'][5:]))
    demos = [load_demo(Path(e['labels']).parent, e['labels_sha256']) for e in entries]
    assert len(demos) == 47
    for entry in entries:
        bound = next(r for r in audit['results'] if r['suite']=='libero_spatial' and r['task_id']==0
                     and r['demo']==entry['job']['demo'])
        folder = Path(entry['labels']).parent
        assert hashlib.sha256((folder/'prepared_input.json').read_bytes()).hexdigest()==bound['prepared_input_sha256']
        assert bound['labels_sha256']==entry['labels_sha256']
    caps = np.array([d['caps_deg'] for d in demos])
    peaks = np.array([d['actual_peak_deg'] for d in demos])
    distribution = {u:dict(search=stats(caps[:, j], UPPER[j]), actual=stats(peaks[:, j]))
                    for j, u in enumerate(UNITS)}
    old_demos = [load_demo(OLD/'libero_spatial/task_00'/f'demo_{i}') for i in range(5)]
    new_first5 = np.array([d['caps_deg'] for d in demos[:5]])
    old_first5 = np.array([d['caps_deg'] for d in old_demos])

    # Decode object free-joint coordinates from the original XML joint order.
    # MuJoCo flattened state is [time, qpos, qvel, ...]. No environment is instantiated.
    scene = {}
    with h5py.File(H5, 'r') as h5:
        doc = ET.fromstring(h5['data/demo_0'].attrs['model_file'])
        qadr = 0
        addresses = {}
        for body in doc.find('worldbody').iter('body'):
            for joint in list(body):
                if joint.tag not in ['joint', 'freejoint']:
                    continue
                kind = 'free' if joint.tag=='freejoint' else joint.get('type', 'hinge')
                if kind == 'free':
                    addresses[body.get('name')] = qadr
                qadr += dict(free=7, ball=4, hinge=1, slide=1)[kind]
        scene_samples = []
        for d in demos:
            g = h5['data'][d['demo']]
            state = g.attrs['init_state']
            assert np.allclose(state[:1+qadr], g['states'][0, :1+qadr])
            positions = {name:state[1+a:1+a+3].tolist() for name,a in addresses.items()}
            scene_samples.append(dict(demo=d['demo'], positions_m=positions))
        for name in addresses:
            x = np.array([d['positions_m'][name] for d in scene_samples])
            distances = np.linalg.norm(x[:,None]-x[None,:],axis=2)[np.triu_indices(len(x),1)]
            scene[name] = dict(axis_std_mm=(x.std(axis=0)*1000).tolist(),
                               axis_span_mm=(np.ptp(x,axis=0)*1000).tolist(),
                               pairwise_distance_mean_mm=float(distances.mean()*1000),
                               pairwise_distance_max_mm=float(distances.max()*1000))
        original_count = len(h5['data'])

    gains = [100*(d['full_velocity']-d['best_velocity'])/d['full_velocity'] for d in demos if d['full_pass']]
    summary = dict(task='libero_spatial/00', original_demos=original_count, analyzed_demos=len(demos),
                   missing_original_demo_ids=sorted(set(range(original_count))-{int(d['demo'][5:]) for d in demos}),
                   directions=distribution, scene_initial_positions=scene,
                   exact_full_vectors=sum(d['full_caps'] for d in demos),
                   near_zero_entries=int(np.isclose(caps,0,atol=1e-6,rtol=0).sum()),
                   upper_entries=int(np.isclose(caps,UPPER,atol=1e-6,rtol=0).sum()),
                   total_releasable_entries=int(caps.size),
                   mean_direction_span_deg=float(np.ptp(caps,axis=0).mean()),
                   mean_direction_pairwise_diff_deg=float(np.mean([v['search']['pairwise_abs_mean'] for v in distribution.values()])),
                   mean_actual_direction_span_deg=float(np.ptp(peaks,axis=0).mean()),
                   mean_actual_pairwise_diff_deg=float(np.mean([v['actual']['pairwise_abs_mean'] for v in distribution.values()])),
                   old_new_first5_abs_diff_deg=(np.abs(new_first5-old_first5)).tolist(),
                   old_new_first5_mean_abs_diff_deg=float(np.abs(new_first5-old_first5).mean()),
                   old_new_first5_max_abs_diff_deg=float(np.abs(new_first5-old_first5).max()),
                   full_pass_count=len(gains), best_vs_full_speed_gain_pct=gains,
                   gain_le_0_1_pct_count=sum(g<=.1 for g in gains),
                   gain_le_1_pct_count=sum(g<=1 for g in gains),
                   frame_count_range=[min(d['frames'] for d in demos),max(d['frames'] for d in demos)])
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (HERE/'details.json').write_text(json.dumps(dict(units=UNITS,demos=demos,old_first5=old_demos,scene=scene_samples),indent=2)+'\n')
    (HERE/'sources.json').write_text(json.dumps(dict(files=evidence,original_hdf5=str(H5),
         annotation_root=str(ROOT),old_root=str(OLD)),indent=2)+'\n')
    with (HERE/'per_demo.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['demo']+[u+' cap_deg' for u in UNITS]+[u+' peak_deg' for u in UNITS])
        for d in demos:w.writerow([d['demo']]+d['caps_deg']+d['actual_peak_deg'])

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
    for j,(ax,u) in enumerate(zip(axes.flat,UNITS)):
        bins=np.linspace(0,UPPER[j],7)
        ax.hist(caps[:,j],bins=bins,color='#376bad',edgecolor='white')
        ax.set(title=u,xlim=(-.6,UPPER[j]+.6),ylim=(0,47),xlabel='Selected boundary (deg)',ylabel='Demo count')
        ax.text(.04,.93,f"upper: {distribution[u]['search']['upper_count']}/47",transform=ax.transAxes,va='top')
    fig.suptitle('Spatial 00 | all 47 labeled demos | concentration at search boundaries',fontsize=14)
    fig.savefig(HERE/'distribution.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(14,6),layout='constrained')
    for ax,m,title in zip(axes,[caps.T,peaks.T],['Selected tolerance boundary (deg)','Realized directional peak in selected offline replay (deg)']):
        im=ax.imshow(m,aspect='auto',cmap='viridis',vmin=0,vmax=30,interpolation='nearest')
        ax.set_yticks(range(8),UNITS);ax.set_xticks(range(len(demos)),[d['demo'][5:] for d in demos],fontsize=7)
        ax.set(title=title,xlabel='Original demo ID');fig.colorbar(im,ax=ax,pad=.01)
    fig.savefig(HERE/'heatmap.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
