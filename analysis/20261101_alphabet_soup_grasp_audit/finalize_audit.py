"""Validate diagnostic artifacts; assemble saved-video evidence and report."""
import csv,io,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/media/czl/sata/franka_my_code/openpi');sys.path.insert(0,str(ROOT))
from experiments.libero_tasktol_compat import diagnose_alphabet_soup_grasps as audit
OUT=audit.OUT;RUN=audit.RUN;AB=OUT/'axis_ablation'
j=json.loads((OUT/'diagnostics.json').read_text());new=json.loads((AB/'full10_summary.json').read_text());assert new['complete']and len(new['results'])==22
variants={mask:[]for mask in ['000','011','001','010']};details=[]
plan=json.loads((RUN/'plan.json').read_text());entries={e['key']:e for e in plan['entries']}
for e in j['episodes']:
 key=e['key'];base=Path(entries[key]['baseline_folder']);b=json.loads((base/'trajectory.json').read_text());t=json.loads((RUN/'episodes'/key/'trajectory.json').read_text())
 np.testing.assert_array_equal([r['action']for r in b],[r['action']for r in t])
 np.testing.assert_allclose([r['nominal_target']['position_m']for r in b],[r['nominal_target']['position_m']for r in t],atol=1e-12,rtol=0)
 c=e['baseline']['close_frame'];summ=dict(key=key,close_frame=c,baseline_success=e['baseline']['success'],tolerance_success=e['tolerance']['success'],**e['close_comparison'],
  baseline_attempts=e['baseline']['attempts'],tolerance_attempts=e['tolerance']['attempts'],
  tolerance_first_robot_contact_frame=e['tolerance']['first_robot_contact_frame'],
  baseline_first_robot_contact_frame=e['baseline']['first_robot_contact_frame'],
  preclose_object_movement_mm=e['tolerance']['preclose_object_movement_mm'])
 for mask,folder in [('000',base),('011',RUN/'episodes'/key)]+[(m,AB/m/key/'joint_native/seed_00/mask30')for m in ['001','010']]:
  r=json.loads((folder/'report.json').read_text());rows=json.loads((folder/'trajectory.json').read_text());variants[mask].append(r['success'])
  summ[f'{mask}_success']=r['success'];summ[f'{mask}_rejected']=r['execution_rejected']
  np.testing.assert_array_equal([z['action']for z in rows],[z['action']for z in b])
  np.testing.assert_allclose([z['nominal_target']['position_m']for z in rows],[z['nominal_target']['position_m']for z in b],atol=1e-12,rtol=0)
  assert all(z['physics_steps']==25 and not z['aborted']for z in rows)
  for k in ['actions_sha256','initial_state_sha256','model_sha256','solver_settings','objective_settings','shared_joint_servo']:
   assert r[k]==json.loads((base/'report.json').read_text())[k]
  summ[f'{mask}_max_ik_position_residual_um']=max(z['optimizer']['position_residual_m']for z in rows)*1e6
  if mask in ['001','010']:
   decode=subprocess.run(['ffmpeg','-v','error','-i',str(folder/'main_rgb.mp4'),'-f','null','-'],capture_output=True);assert decode.returncode==0 and not decode.stderr
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(folder/'main_rgb.mp4')]))['streams'][0]
   assert int(probe['nb_read_frames'])==len(b)and probe['r_frame_rate']=='20/1'
 details.append(summ)
manifest=json.loads((AB/'source_snapshot_manifest.json').read_text())
for x in manifest:assert hashlib.sha256((AB/x['snapshot']).read_bytes()).hexdigest()==x['sha256']
counts={m:dict(success=sum(v),total=len(v))for m,v in variants.items()}
validation=dict(complete=True,counts=counts,additional_axis_rollouts=20,original011_exact_reproduction_checks=2,
 all_actions_positions_initial_states_models_optimizer_servo_unchanged=True,all20_new_videos_decoded=True,
 all20_new_videos_20hz_full_length=True,all_saved_snapshot_hashes_checked=len(manifest),
 all_ablation_rollouts_no_solver_rejections=True,episodes=details,
 interpretation='Fixed ten previously reconstructed successful demos. Diagnostic ablation: changing PRE mask only, EXEC111 retained. No claim about unseen seeds or other tasks. State contacts reconstructed geometrically at20Hz; contact forces not inferred.')
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
# Two synchronized moments of the same demo, drawn only from saved native videos.
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,17);small=ImageFont.truetype(fontpath,14)
canvas=Image.new('RGB',(4*320,2*380+58),(13,22,30));draw=ImageDraw.Draw(canvas)
draw.text((12,10),'Same demo_4 / 20Hz / original xyz + gripper / same IPOPT and native servo',font=font,fill='white')
cols=[('000','No tolerance',Path(entries['libero_object/00/demo_4']['baseline_folder'])),('011','PRE011: yF + zF',RUN/'episodes/libero_object/00/demo_4'),('001','PRE001: zF only',AB/'001/libero_object/00/demo_4/joint_native/seed_00/mask30'),('010','PRE010: yF only',AB/'010/libero_object/00/demo_4/joint_native/seed_00/mask30')]
for rownum,frame in enumerate([54,80]):
 for col,(mask,label,folder)in enumerate(cols):
  png=subprocess.check_output(['ffmpeg','-v','error','-i',str(folder/'main_rgb.mp4'),'-vf',f'select=eq(n\\,{frame})','-frames:v','1','-f','image2pipe','-vcodec','png','-'])
  img=Image.open(io.BytesIO(png)).convert('RGB');target=OUT/'frames'/f'demo4_pre{mask}_f{frame}.png';target.parent.mkdir(exist_ok=True);img.save(target)
  x,y=col*320,58+rownum*380;canvas.paste(img.resize((320,320)),(x,y+28));draw.text((x+8,y+2),label,font=font,fill='white')
  rows=json.loads((folder/'trajectory.json').read_text());rep=json.loads((folder/'report.json').read_text());rr=rows[frame]
  draw.text((x+8,y+350),f'f{frame} ({frame/20:.2f}s) | final '+('PASS'if rep['success']else'FAIL'),font=small,fill='#59e3be'if rep['success']else'#ff9696')
canvas.save(OUT/'demo4_axis_comparison.png')
# Main report: auditable findings, no private reasoning traces.
lines=['# Alphabet soup：抓取容差失效诊断（2026-11-01）','',
'结论：该固定重构位置路径下，PRE011中的zF偏航自由度是主要失效因素。全10条控制对照：PRE011为2/10成功，PRE001为2/10，PRE010为10/10。不是IPOPT求解拒绝；改变的是PRE语义mask，未修改优化器、动作、速度、夹爪、伺服或物理环境。正式400条结果和mask配置未覆盖。','',
'## 对照设计','',
'原无容差10条，原PRE011/EXEC111十条；新增PRE001/EXEC111与PRE010/EXEC111各十条。ENG/DIS000、上限30°、20Hz动作/500Hz物理。两条011原样复测的完整states和500Hz servo_physics数组逐元素完全一致。每条新视频已完整解码并核对帧数。','',
'|demo|无容差|原011|仅zF 001|仅yF 010|闭爪帧EE差/mm|闭爪帧物体差/mm|EE姿态差/°|',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for e in details:lines.append(f"|{e['key'].split('/')[-1]}|{int(e['000_success'])}|{int(e['011_success'])}|{int(e['001_success'])}|{int(e['010_success'])}|{e['ee_separation_mm']:.2f}|{e['object_separation_mm']:.2f}|{e['ee_rotation_separation_deg']:.2f}|")
lines+=['','## 位置与接触证据','',
'所有原无容差/原011/新001/新010：输入7维动作完全相同，各帧名义位置相同（1e-12m校验），初始state、模型、默认IPOPT设置/目标和共享伺服参数一致。IK位置残差与实际跟踪误差分开统计，见validation.json。','',
'原011失败8条中，demo2/4/5/10/18/30/33在首次闭爪前1–5帧已出现单指与汤罐接触；相对无容差最早接触提前3–8帧（0.15–0.40秒）。抓取前指令尚为打开；接触后实际末端落后于名义位置，单指挡在罐体上沿附近，不能形成持续搬运。没有在这个时间窗发现手臂/夹爪接触其他场景物体或桌面，接触对象是目标汤罐。','',
'demo4：原011第45帧（2.25s）左指碰罐，第49帧（2.45s）才闭爪；无容差第51帧才接触。第49帧物体相对无容差已偏11.69mm，实际末端差37.40mm，姿态差30.33°。第54帧无容差末端高度约47.0mm，原011约72.4mm，夹爪被挡在更高处。原011闭爪前位移约11.70mm；整个首次闭爪期间仅2帧双指几何接触，最大抬升15.7mm。','',
'demo2说明物体先被推偏不是必要条件：首次闭爪时物体与无容差仅差0.56mm，但EE已经相差42.74mm，单指提前碰到罐体并阻碍继续下降。','',
'demo29必须单列：无容差也存在第48–58帧首次抓空，原示范第78帧有第二次抓取。无容差第二次抬升233.0mm而成功，原011第二次仅抬升10.5mm。不能把第一段抓空单独归因于容差。','',
'demo33有一次约140mm抬升后失去搬运，属于暂时夹住后失稳；其他7条原011失败轨迹的各次闭爪抬升峰值均不超过约16mm。','',
'## 机制解释与边界','',
'1. 当前是固定TCP位置轨迹、旋转可放宽；原路径中的TCP并不始终位于罐轴/抓取中心。绕TCP偏航与绕物体轴偏航不是同一刚体运动。物体具有近似旋转对称性，也不能推出沿任意固定位置接近路径放开30°偏航仍能把两指放到物体两侧。姿态改变影响整个手指扫掠体积和接触顺序。','',
'2. 接近时对约64–69mm投影宽度的罐体，全开夹爪内间隙约78–80mm，居中时每侧仅约5–8mm余量。这个全局投影仅是包络线索，不是完整接触/力闭合判据。原011失败案例的提前接触帧，物体对夹爪中线的偏置常超过该余量，且罐中心仍在指尖下方；先碰上沿，随后实际位姿误差放大。','',
'3. ENG000的含义是相对阶段交接姿态不再新增释放，不是恢复独立原名义姿态。release_state.begin_stage保留optimized_rotation；所以ENG行rotation_release_rad为零时，对原名义姿态仍可偏约23–32°。这与共享MuJoCo既有语义一致。','',
'4. 默认IPOPT约束中未加入该罐体/手指接触可行性验证；它可以精确满足位置与允许姿态约束，却选到这条接近路径无法实现有效抓取的姿态。所有对照不改它。几何接触由保存state重建，未把重建的接触力当作录制力；误差与接触时刻一致，加上轴向闭环对照，支持上述因果机制。','',
'5. 位置=gripper0_grip_site，旋转=robot0_right_hand；site相对手部有固定约90°yaw偏置。分析已按手部旋转及实际指节运动轴校准，未将site的X/Y轴错认。XML固定四元数序列化造成的亚微弧度差已显式容忍；所有state位置FK核验到1e-10m。','',
'## 文献对应','',
'[Berenson et al., Task Space Regions, IJRR 2011](https://publications.ri.cmu.edu/storage/publications/pub_files/2011/10/dmitry_ijrr10-1.pdf)：第4节用任务参考系T0_w与末端偏置Tw_e描述SE(3)允许集合，图4包含汽水罐抓取；物体/抓取点的自由度需要通过偏置映射为末端位姿，不能把物体对称性直接等同于固定TCP旋转盒。第1/4节另将碰撞和任务位姿约束区分。这里是对该框架的应用解释，不声称论文研究了此LIBERO实例。','',
'[MuJoCo3.2.3 Computation](https://mujoco.readthedocs.io/en/3.2.3/computation/)：接触约束产生动力学力，位置伺服参考不等于实际状态；接触几何和控制器期望应分别检查。本诊断遵循这一区分。','',
'## 建议','',
'当前最小修改候选为该任务PRE011→010，POST111保留；固定10条已验证10/10，但尚未更新正式mask或viewer。若保留zF自由度，则需在进入接触前恢复名义偏航，或围绕物体/抓取中心同时变换位置轨迹并验证手指接近包络；不应在闭爪之后才恢复，也无需修改IPOPT。后续泛化需新示例验证。','',
'## 留存','',
'诊断脚本：experiments/libero_tasktol_compat/diagnose_alphabet_soup_grasps.py。diagnostics.json为20条原配对轨迹全帧对象/EE位姿与接触几何；summary.csv为原配对摘要；validation.json为动作/名义位置/视频/快照验证；axis_ablation内保留新增全部20条rollout、2条原样复测及执行源码快照。frames和demo4_axis_comparison.png为原视频截帧。没有修改正式数据。']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(counts=counts,verified_new_videos=20,snapshot_files=len(manifest),output=str(OUT)),ensure_ascii=False))
