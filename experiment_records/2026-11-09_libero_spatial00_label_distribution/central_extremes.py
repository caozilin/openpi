"""Median, concentration and unusual demos from the existing audited dataset."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
DETAILS = HERE/'details.json'
document = json.loads(DETAILS.read_text())
units = document['units']
demos = document['demos']
upper = np.array([30., 30., 15., 15., 15., 15., 15., 15.])
result = dict(n=47, units=units, upper_deg=upper.tolist(),
              definitions=dict(center='Coordinate-wise median across all 47 demos',
                  central_interval='Q25 to Q75, NumPy linear quantile interpolation',
                  neighborhood='Absolute deviation from coordinate median <=5 degrees, with 1e-8 degree numerical tolerance',
                  extremeness='Mean over 8 coordinates of absolute deviation from coordinate median divided by authored upper bound (30 APP, 15 EXEC)',
                  modal_value='Most frequent value after rounding to 6 decimal degrees; report all ties',
                  actual='Maximum directional release in each phase of the selected OFFLINE candidate; not the subsequent physical rollout or SO(3) peak'),
              inputs=[dict(path=str(DETAILS),sha256=hashlib.sha256(DETAILS.read_bytes()).hexdigest())])
physical_root = Path('/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/episodes/libero_spatial/00')
result['physical_report_seed_by_demo'] = {}
for demo in demos:
    path = physical_root/demo['demo']/'report.json'
    content = path.read_bytes()
    report = json.loads(content)
    result['physical_report_seed_by_demo'][demo['demo']] = report['seed']
    result['inputs'].append(dict(path=str(path),sha256=hashlib.sha256(content).hexdigest()))
assert set(result['physical_report_seed_by_demo'].values())=={0}

for kind, field in [('annotation','caps_deg'),('actual','actual_peak_deg')]:
    x = np.array([d[field] for d in demos])
    med = np.median(x,axis=0)
    low, high = np.quantile(x,[.25,.75],axis=0)
    abs_deviation = np.abs(x-med)
    normalized = (abs_deviation/upper).mean(axis=1)
    ordered = sorted(range(47),key=lambda i:(-normalized[i],int(demos[i]['demo'][5:])))
    nearest = min(range(47),key=lambda i:(normalized[i],int(demos[i]['demo'][5:])))
    marginal=[]
    for j, unit in enumerate(units):
        counts = Counter(np.round(x[:,j],6).tolist())
        max_count = max(counts.values())
        marginal.append(dict(unit=unit, median_deg=float(med[j]),q25_deg=float(low[j]),
             q75_deg=float(high[j]),iqr_deg=float(high[j]-low[j]),
             iqr_over_authored_upper=float((high[j]-low[j])/upper[j]),
             within5_count=int((abs_deviation[:,j]<=5+1e-8).sum()),
             within5_fraction=float((abs_deviation[:,j]<=5+1e-8).mean()),
             modes_deg=sorted(v for v,n in counts.items() if n==max_count),mode_count=max_count))
    def row(i):
        d=demos[i]
        return dict(demo=d['demo'],normalized_mean_absolute_deviation=float(normalized[i]),
            mean_absolute_deviation_deg=float(abs_deviation[i].mean()),
            caps_deg=d['caps_deg'],actual_peak_deg=d['actual_peak_deg'])
    result[kind] = dict(marginals=marginal, median_vector_deg=med.tolist(),
         mean_normalized_iqr=float(((high-low)/upper).mean()),
         within5_coordinate_count=int((abs_deviation<=5+1e-8).sum()),
         coordinate_count=int(x.size),
         within5_all_coordinates_count=int((abs_deviation<=5+1e-8).all(axis=1).sum()),
         median_of_per_demo_maximum_direction_deg=float(np.median(x.max(axis=1))),
         top5=[row(i) for i in ordered[:5]], nearest_to_median=row(nearest),
         ranking=[row(i) for i in ordered])
    unique=[]
    seen=set()
    for i in ordered:
        signature=tuple(np.round(x[i],6))
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(row(i))
        if len(unique)==5:
            break
    result[kind]['top5_distinct_vectors']=unique

# Independently recompute scores with Python arithmetic and quantiles with
# interpolation in the sorted observations; include rank order and tie handling.
for kind, field in [('annotation','caps_deg'),('actual','actual_peak_deg')]:
    r=result[kind]
    centers=[statistics.median(d[field][j] for d in demos) for j in range(8)]
    scores={}
    for d in demos:
        scores[d['demo']]=statistics.fmean(abs(d[field][j]-centers[j])/upper[j] for j in range(8))
    expected=sorted(scores,key=lambda k:(-scores[k],int(k[5:])))
    assert expected==[v['demo'] for v in r['ranking']]
    for j, m in enumerate(r['marginals']):
        vals=sorted(d[field][j] for d in demos)
        for p, key in [(.25,'q25_deg'),(.75,'q75_deg')]:
            index=(len(vals)-1)*p
            i=math.floor(index);v=vals[i]+(vals[math.ceil(index)]-vals[i])*(index-i)
            assert math.isclose(v,m[key],abs_tol=1e-10)
        assert sum(abs(v-centers[j])<=5+1e-8 for v in vals)==m['within5_count']
    for v in r['ranking']:
        assert math.isclose(v['normalized_mean_absolute_deviation'],scores[v['demo']],abs_tol=1e-12)

(HERE/'central_extremes.json').write_text(json.dumps(result,indent=2)+'\n')
with (HERE/'extreme_demos.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['ranked_by','rank','demo','normalized_mean_abs_deviation']+
                              [u+' cap_deg' for u in units]+[u+' actual_peak_deg' for u in units])
    for kind in ['annotation','actual']:
        for rank, row in enumerate(result[kind]['top5'],1):
            w.writerow([kind,rank,row['demo'],row['normalized_mean_absolute_deviation']]+row['caps_deg']+row['actual_peak_deg'])

lines=['## 中位数、中央集中度与极端样本（2026-11-09续分析）','',
 '仍使用上文全部47条同协议demo；不增加搜索或回放。每方向中位数为中心，中央区间为Q25–Q75。±5°占比表示47条中落在该方向中位数±5°内的数量。各方向合成的中位数向量不保证在某条真实demo中同时出现。实际释放沿用选中候选的离线方向峰值。','',
 '|方向|标注中位数 [Q25,Q75]°|标注±5°|实际峰值中位数 [Q25,Q75]°|实际±5°|',
 '|---|---:|---:|---:|---:|']
for a,b in zip(result['annotation']['marginals'],result['actual']['marginals']):
    cell=lambda m:f"{m['median_deg']:.2f} [{m['q25_deg']:.2f}, {m['q75_deg']:.2f}]"
    lines.append(f"|{a['unit']}|{cell(a)}|{a['within5_count']}/47|{cell(b)}|{b['within5_count']}/47|")
lines+=['',
 '中位数和众数分开解释：标注每方向的众数为APP Ry−/Ry+各30°，EXEC六方向各15°；实际峰值众数除EXEC Ry+为15°外，其余七方向均为0°。众数按1e-6度精度归并，具体频数见central_extremes.json。标注的逐方向众数组合没有在这47条中完整同时出现。','']
for kind, title, field in [('annotation','标注边界','caps_deg'),('actual','实际释放峰值','actual_peak_deg')]:
    r=result[kind]
    lines += [f"### 按{title}找偏离中位数最大的5条",'',
       '极端程度D = mean_j(|x_j−median_j| / authored_upper_j)。APP除30°，EXEC除15°；8个方向等权，按D降序，完全并列按demo编号升序。该排名衡量偏离典型向量的程度，包含偏大与偏小；不等于按最大单方向数值排序。','',
       '|demo|D|APP Ry[−,+]°|EXEC Rx[−,+]°|EXEC Ry[−,+]°|EXEC Rz[−,+]°|',
       '|---|---:|---:|---:|---:|---:|']
    for row in r['top5']:
        vals=row[field]
        cells=[' / '.join(f'{v:.2f}' for v in vals[i:i+2]) for i in range(0,8,2)]
        lines.append('|'+row['demo']+f"|{row['normalized_mean_absolute_deviation']:.4f}|"+'|'.join(cells)+'|')
    lines += ['',f"各方向±5°总覆盖{r['within5_coordinate_count']}/376；全部8方向同时在±5°内{r['within5_all_coordinates_count']}/47。IQR除以该方向语义上限后取8方向平均为{r['mean_normalized_iqr']*100:.2f}%。最接近中位数向量的真实样本是{r['nearest_to_median']['demo']}。",'',
              '若希望5种不同向量，按相同极端度排序、将1e-6度精度下完全相同的向量只保留第一条，则为：'+
              '、'.join(v['demo'] for v in r['top5_distinct_vectors'])+'。这一步仅去除重复向量，并未声称5条之间距离全局最大。','']
lines += ['若强行压成每条demo的单一最大方向值，再对47条取中位数，标注为30.00°，实际峰值为14.59°。这会隐藏方向及正负不对称，主结果保留8维。', '',
 '编号核查：后续物理回放report.json的seed字段固定为0，不能把demo_44写成seed44；这里使用原始演示编号。各demo的初态与轨迹确实不同，编号不是独立BO重复运行次数。','',
 '追加统计文件绝对路径：`'+str(HERE/'central_extremes.json')+'`、`'+str(HERE/'extreme_demos.csv')+'`、`'+str(HERE/'central_extremes.py')+'`、`'+str(HERE/'central_extremes_verification.json')+'`。原始来源沿用上文sources.json；输入details.json的SHA另在central_extremes.json绑定。原始候选中demo44、45、49均选中零边界，不将它们当作缺失值。', '']
(HERE/'CENTRAL_EXTREMES.md').write_text('\n'.join(lines))
(HERE/'central_extremes_verification.json').write_text(json.dumps(dict(complete=True,n=47,
    directions=8,verified=['median','linear quartiles','within5 count','normalized L1 score','all 47 rank order'],
    input=result['inputs'],new_search_or_simulation=False),indent=2)+'\n')
for kind in ['annotation','actual']:
    print(kind,json.dumps({k:v for k,v in result[kind].items() if k not in ['ranking','marginals']},indent=2))
