"""Merge explicit offline searches, preserving all trials and acceptance costs."""

import argparse
import json
from pathlib import Path

from closed_loop_state_metrics import evaluate_run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--refinement',type=Path,required=True)
    parser.add_argument('--searches',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=json.loads((args.refinement/'summary.json').read_text())
    assert result['complete']
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    by_key={(r['case'],r['demo']):r for r in result['results']}
    for search in args.searches:
        data=json.loads((search/'summary.json').read_text());assert data['complete']
        parent=json.loads((Path(data['baseline_root'])/'provenance.json').read_text())['plan'][0]
        row=by_key[(parent['case'],parent['demo'])]
        assert Path(data['baseline_root']).resolve()==Path(row['selected_root']).resolve()
        for candidate in data['results']:
            metrics=evaluate_run(candidate['root'])
            accepted=(metrics['full_length'] and not metrics['rejected']
                and (not row['success'] or metrics['success']) and metrics['objective']<row['objective'])
            row['trials'].append(dict(root=candidate['root'],parent=data['baseline_root'],method='CEM',
                parameters=candidate['parameters'],previous_objective=row['objective'],objective=metrics['objective'],
                previous_success=row['success'],success=metrics['success'],accepted=accepted))
            if accepted:
                row.update(selected_root=candidate['root'],objective=metrics['objective'],success=metrics['success'])
    result.update(success=sum(r['success'] for r in result['results']),
        final_mean_cost=sum(r['objective'] for r in result['results'])/len(result['results']),
        policy='policy.json', offline_trial_count=sum(len(r['trials']) for r in result['results']))
    policy=dict(base_policy=str((args.refinement/'policy.json').resolve()),
        sources=['https://lup.lub.lu.se/search/files/206302733/IROS24.pdf','https://arxiv.org/html/2502.20382v2','https://arxiv.org/html/2212.00541'],
        searches=[str(p.resolve()) for p in args.searches],
        scope='IRLC two-step rule applied to all 11 episodes; additional fixed-budget CEM tested on two diagnostic episodes, not an all-task CEM benchmark',
        acceptance='full feasible physical rollout; retain existing task success; strictly decrease fixed complete-state objective',
        time_scale=1,gripper='original unchanged',online_future_or_joint_witness=False)
    (output/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))


if __name__=='__main__':
    main()
