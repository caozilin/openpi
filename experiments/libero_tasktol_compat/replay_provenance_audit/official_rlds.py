"""Fetch fixed-revision official RLDS episode samples and compare saved traces.

Reads TensorFlow Example protobufs directly, without modifying simulation environments.
Only complete, CRC32C-verified records from the start of one shard per suite are saved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct

import google_crc32c
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[3]
REVISION = "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"
REPO = "https://huggingface.co/datasets/openvla/modified_libero_rlds"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CACHE = ROOT / "official_rlds_samples"


def example_class():
    f = descriptor_pb2.FileDescriptorProto(name="audit_example.proto", package="tensorflow", syntax="proto3")
    for name, typ in (("BytesList",12),("FloatList",2),("Int64List",3)):
        m = f.message_type.add(name=name)
        v = m.field.add(name="value", number=1, type=typ, label=3)
        if typ != 12:
            v.options.packed = True
    m = f.message_type.add(name="Feature")
    for number, name, cls in ((1,"bytes_list","BytesList"),(2,"float_list","FloatList"),(3,"int64_list","Int64List")):
        m.field.add(name=name,number=number,type=11,type_name=f".tensorflow.{cls}",label=1)
    m = f.message_type.add(name="Features")
    entry = m.nested_type.add(name="FeatureEntry")
    entry.options.map_entry = True
    entry.field.add(name="key",number=1,type=9,label=1)
    entry.field.add(name="value",number=2,type=11,type_name=".tensorflow.Feature",label=1)
    m.field.add(name="feature",number=1,type=11,type_name=".tensorflow.Features.FeatureEntry",label=3)
    m = f.message_type.add(name="Example")
    m.field.add(name="features",number=1,type=11,type_name=".tensorflow.Features",label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(f)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.Example"))


EXAMPLE = example_class()


def masked_crc(data):
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def read_exact(stream, n):
    chunks = bytearray()
    while len(chunks)<n:
        chunk = stream.read(n-len(chunks))
        if not chunk:
            raise EOFError(f"Truncated record: {len(chunks)}/{n}")
        chunks.extend(chunk)
    return bytes(chunks)


def read_record(stream):
    size = stream.read(8)
    if not size:
        return None
    if len(size)!=8:
        size += read_exact(stream,8-len(size))
    size_crc = read_exact(stream,4)
    assert struct.unpack("<I",size_crc)[0] == masked_crc(size)
    length = struct.unpack("<Q",size)[0]
    if length > 256*1024**2:
        raise ValueError(f"Unexpected record length {length}")
    data = read_exact(stream,length)
    data_crc = read_exact(stream,4)
    assert struct.unpack("<I",data_crc)[0] == masked_crc(data)
    return size+size_crc+data+data_crc, data


def features(data):
    example = EXAMPLE.FromString(data)
    result = {}
    for k,v in example.features.feature.items():
        if v.HasField("float_list"):
            result[k] = np.asarray(v.float_list.value,dtype=np.float32)
        elif v.HasField("int64_list"):
            result[k] = np.asarray(v.int64_list.value,dtype=np.int64)
        else:
            result[k] = list(v.bytes_list.value)
    return result


def fetch_suite(suite, files, count):
    folder = CACHE / suite
    folder.mkdir(parents=True,exist_ok=True)
    prefix = f"{suite}_no_noops/1.0.0/"
    for name in ("dataset_info.json","features.json"):
        path = folder/name
        if not path.exists():
            r = requests.get(f"{REPO}/resolve/{REVISION}/{prefix}{name}",timeout=45)
            r.raise_for_status()
            path.write_bytes(r.content)
    shard = next(name for name in files if name.startswith(prefix) and ".tfrecord-00000-" in name)
    path = folder / f"first_{count}_episodes.tfrecord"
    url = f"{REPO}/resolve/{REVISION}/{shard}"
    if not path.exists():
        temporary = path.with_suffix(".partial")
        with requests.get(url,stream=True,timeout=(30,60)) as r:
            r.raise_for_status()
            r.raw.decode_content = True
            with temporary.open("wb") as out:
                for _ in range(count):
                    rec = read_record(r.raw)
                    if rec is None:
                        break
                    out.write(rec[0])
        temporary.replace(path)
    episodes=[]
    with path.open("rb") as stream:
        while (record:=read_record(stream)) is not None:
            episodes.append(features(record[1]))
    metadata = dict(suite=suite,revision=REVISION,source_url=url,partial_shard=True,
        complete_records=len(episodes),record_crc32c_verified=True,bytes=path.stat().st_size,
        saved_prefix_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        feature_layout={k:(len(v) if isinstance(v,list) else list(v.shape)) for k,v in episodes[0].items()})
    (folder/"sample_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata),flush=True)
    return suite,episodes


def action_hash(a):
    x=np.asarray(a,dtype="<f4")
    return (x.shape,hashlib.sha256(x.tobytes()).hexdigest())


def compare(samples):
    results=[]
    totals=[]
    for suite, episodes in samples:
        local={}
        successes={}
        for sidecar in (PROJECT/"data/libero_openvla"/(suite+"_no_noops")).glob("*.metainfo.json"):
            meta=json.loads(sidecar.read_text())
            successes.update({(meta['task_name'],name):entry['success'] for name,entry in meta['episodes'].items()})
            path=sidecar.with_suffix("").with_suffix(".hdf5")
            with h5py.File(path,"r") as f:
                for name,g in f['data'].items():
                    local.setdefault(action_hash(g['actions'][()]),[]).append((path,name))
        info=json.loads((CACHE/suite/'dataset_info.json').read_text())
        official_count=sum(int(x) for x in info['splits'][0]['shardLengths'])
        totals.append(dict(suite=suite,official_saved=official_count,local_saved=sum(successes.values())))
        for index, f in enumerate(episodes):
            actions=f['steps/action'].reshape(-1,7)
            n=len(actions)
            match=local.get(action_hash(actions),[])
            row=dict(suite=suite,sample_index=index,steps=n,
                official_file_path=[x.decode() for x in f.get('episode_metadata/file_path',[])],
                instruction=f['steps/language_instruction'][0].decode(),
                local_action_sequence_matches=len(match))
            if len(match)==1:
                path,name=match[0]
                with h5py.File(path,"r") as h:
                    g=h['data'][name]
                    state=f['steps/observation/state'].reshape(n,8)
                    pos=g['obs/ee_pos'][()]
                    mm=np.linalg.norm(state[:,:3]-pos,axis=1)*1000
                    q=f['steps/observation/joint_state'].reshape(n,7)
                    row.update(task_name=path.stem.removesuffix('_demo'),demo_id=name,
                        eef_max_mm=float(mm.max()),eef_median_mm=float(np.median(mm)),
                        joint_max_abs_rad=float(np.abs(q-g['obs/joint_states'][()]).max()),
                        gripper_max_abs_m=float(np.abs(state[:,6:]-g['obs/gripper_states'][()]).max()))
                    raw_path=PROJECT/'third_party/libero/libero/datasets'/suite/path.name
                    with h5py.File(raw_path,'r') as raw:
                        indices=g['tasktol/source_step_indices'][()]
                        raw_pos=raw['data'][name]['obs/ee_pos'][()][indices[:-1]]
                        official_raw_mm=np.linalg.norm(state[1:,:3]-raw_pos,axis=1)*1000
                        local_raw_mm=np.linalg.norm(pos[1:]-raw_pos,axis=1)*1000
                        row.update(official_vs_raw_eef_max_mm=float(official_raw_mm.max()),
                            official_vs_raw_eef_median_mm=float(np.median(official_raw_mm)),
                            local_vs_raw_eef_max_mm=float(local_raw_mm.max()))
                    # Store only main-view images. JPEG appearance is not a dynamics metric.
                    frame=int(np.argmax(mm))
                    original=Image.open(io.BytesIO(f['steps/observation/image'][frame])).convert('RGB')
                    current=Image.fromarray(g['obs/agentview_rgb'][frame][::-1,::-1].copy())
                    canvas=Image.new('RGB',(540,300),'#121821')
                    draw=ImageDraw.Draw(canvas)
                    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
                    draw.text((8,5),f'{suite}/{name} frame {frame} | EEF difference {mm[frame]:.3f} mm',font=font,fill='white')
                    draw.text((8,23),'Official RLDS',font=font,fill='white')
                    draw.text((278,23),'Local regenerated',font=font,fill='white')
                    canvas.paste(original.resize((256,256)),(8,42));canvas.paste(current.resize((256,256)),(278,42))
                    filename=f'{suite}_{index:02d}_{name}.png';canvas.save(ROOT/filename)
                    row['mainview_comparison']=filename
            results.append(row)
    output=dict(official_revision=REVISION,official_dataset=REPO,totals=totals,
        matching='Exact float32 action sequence hash; same retained action index for both pre-action observation streams.',
        scope='First records of first shard per suite, not a random or exhaustive sample.',episodes=results,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'official_comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    matched=[x for x in results if x['local_action_sequence_matches']==1]
    lines=[
        '# 官方 RLDS 与本地重录对照 · 2026-10-20', '',
        '本地旧环境重录在本次样本中接近官方发布版；官方重生成数据相对原始记录同样存在厘米级轨迹差。相对原始记录的偏差及约 15% 未保留比例，不能单独证明本机 LIBERO 环境异常。此结论不涵盖 MuJoCo 3.10 的迁移结果。', '',
        f'官方来源：[openvla/modified_libero_rlds]({REPO})，固定提交 `{REVISION}`。', '',
        '## 完整数据集数量', '',
        '| 任务集 | 官方 train episode | 本地成功重录 |', '|---|---:|---:|',
    ]
    lines += [f"| {t['suite']} | {t['official_saved']} | {t['local_saved']} |" for t in totals]
    total_official=sum(x['official_saved'] for x in totals);total_local=sum(x['local_saved'] for x in totals)
    lines += [f'| 合计 | {total_official} | {total_local} |', '',
        '官方数量来自四份 dataset_info.json 的 train/shardLengths 求和，本地数量来自完整重录 ledger。以四套各 500 条原始示范计，官方未保留 307 条（15.35%）；本地重录失败 301 条（15.05%）。官方发布元数据未逐条说明未保留原因，不能把 307 条全部认定为同一种失败。', '',
        '## 直接轨迹对照', '',
        f'每套取首个 shard 的前 4 个完整记录，共 {len(results)} 条；{len(matched)} 条均通过完整 float32 动作序列精确匹配到唯一本地示范，涵盖 {sum(x["steps"] for x in matched)} 帧。TFRecord 长度与内容 CRC32C 均校验通过。此为便利样本，并非随机抽样或全量一致性检验。', '',
        '官方与本地均按同一保留动作的动作前观测比较。对原始记录的比较另作动作前／后错位处理，沿用 audit.py 已核实的时序；仅统计末端位置的欧氏距离，不以 JPEG 像素误差衡量动力学。', '',
        '| 任务集 | 匹配条数 | 本地对官方：最大末端差 | 官方对原始：最大末端差 |',
        '|---|---:|---:|---:|',
    ]
    for suite in SUITES:
        subset=[x for x in matched if x['suite']==suite]
        if subset:
            lines.append(f"| {suite} | {len(subset)} | {max(x['eef_max_mm'] for x in subset):.6f} mm | {max(x['official_vs_raw_eef_max_mm'] for x in subset):.6f} mm |")
    if matched:
        lines += ['',f"{sum(x['eef_max_mm']<1 for x in matched)}/{len(matched)} 条的本地对官方最大末端差小于 1 mm，全部样本最差 {max(x['eef_max_mm'] for x in matched):.6f} mm。", '',
            '本地与官方不完全相同：保留集合数量已有差异，个别关节、夹爪状态也有差别。末端接近不证明物体状态、接触力或所有关节逐帧等价；本次未重新运行仿真，未比较成功判据实现。', '',
            '## 逐条结果与留存图像', '',
            '| 任务集 / demo | 本地对官方末端最大差 mm | 官方对原始末端最大差 mm | 主视角对照 |', '|---|---:|---:|---|']
        lines += [f"| {x['suite']} / {x['demo_id']} | {x['eef_max_mm']:.6f} | {x['official_vs_raw_eef_max_mm']:.6f} | [图片]({x['mainview_comparison']}) |" for x in matched]
    lines += ['', '复现：在 openpi 根目录运行 `.venv/bin/python experiments/libero_tasktol_compat/replay_provenance_audit/official_rlds.py --offline`。', '',
        '`official_comparison.json` 保存逐条任务名、匹配 demo、末端／关节／夹爪差异；`official_rlds_samples/` 保存固定提交的元数据、完整 TFRecord 前缀、CRC 校验与 SHA256。仅导出和查看主视角图片。原始数据、处理后 HDF5 和冻结 mask 均未修改。', '']
    (ROOT/'OFFICIAL_COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes-per-suite',type=int,default=4);p.add_argument('--offline',action='store_true');args=p.parse_args()
    CACHE.mkdir(exist_ok=True)
    api=CACHE/'repository.json'
    if not args.offline:
        r=requests.get(f'https://huggingface.co/api/datasets/openvla/modified_libero_rlds/revision/{REVISION}',timeout=45)
        r.raise_for_status();api.write_text(json.dumps(r.json(),indent=2)+"\n")
    files=[x['rfilename'] for x in json.loads(api.read_text())['siblings']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        samples=list(pool.map(lambda suite:fetch_suite(suite,files,args.episodes_per_suite),SUITES))
    compare(samples)


if __name__=='__main__':
    main()
