21:00监督：真实日志到Step130，loss0.6040、raw action0.2715、raw tolerance0.3998；原总控3339995及完成监督3339996存活，未重启。前100步全部10条日志有限，4x恒等式和SwanLab全部对应云端指标逐值核对通过，证据training_prefix100_verification.json。尚无2000checkpoint，更未完成10000；当前结论仅启动和已观察前缀健康。

20:46实测：新版已真正完成至少10步；六张GPU0/1/2/3/4/6全部真实计算，trainer3339998及8worker、总控3339995、最终监督3339996均存活。Step10 loss1.0650/grad_norm2.5589，raw action0.4343、weighted action0.6414、raw tolerance0.8473、weighted tolerance0.4236；批次动作平均权重1.4667，对应容差来源比例0.1556。SwanLab云端相同步10的全部10指标已读取，数值匹配本地，真实4x贡献恒等式通过。证据four_x_protocol_update/first_real_metrics_and_cloud_verification.json与training_live_verified_20261109_2046.json。此仅首10步验证，不能替代完整10000/五checkpoint/1000云端行/最终独立审计，继续复用当前实例，不重启。

## 最新状态：4倍权重正式训练已启动（20:40）

本地与远端数据完整审计已通过；4338batch归一化真实exit0，state8/actions13全部有限，统计SHA `e26cd8d5baade92e2f081435cb497df85f569439fa1e12e4f93fa7b66d728e42`。用户训练前追加4倍容差动作损失，已完成版本化部署与真实LeRobot/归一化/Observation/JIT预检；原统计输入未变、原receipt SHA未改，不重算统计。旧总控3312670与旧结束监督3210109已定向退出，旧请求/源码完整保存至本实验four_x_protocol_update/before_4x，原GPU优化步数0。

新唯一训练总控3339995、训练子3339998、最终监督3339996真实活跃。请求SHA `4c9a7fc7d88dc7c1c07df20c7276a264ed6f9df052cca151318370c8a7268fb0`，实际数据变换tolerance_trajectory_action_loss_weight=4.0；名义动作7维×1、容差观测来源动作7维×4、容差6维组权重0.5。其他batch72/6卡0,1,2,3,4,6/FSDP2/10000steps/warmup333/原LoRA/48tokens/nostate/horizon10不变。原“无4倍”和“统计运行中/父T”段落均为历史，不能据此CONT旧总控或重复启动。

SwanLab新在线实验已通过OpenApi真实查询，云端RUNNING：[kl28cdu6bv1mywr8n5wv0](https://swanlab.cn/@caozilin/openpi/runs/kl28cdu6bv1mywr8n5wv0)。当前处于训练worker/模型加载阶段，首个真实Step指标及云端loss记录仍待产生；不能把上线/启动等同于10000完成。后续继续检查实际8worker、6卡、前10步4x损失分解与云端同步，最终须五checkpoint完整CPU恢复和全1000条云端逐行审计。

本地证据根 `/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/four_x_protocol_update`：transition_completion、original_normalization_exit、prior_protocol_retirement、normalization_reuse_verification、training_preflight_verification、startup_verification、swanlab_online_startup_verification及unit_test_verification（34+6=40回归）。当前源码/实际配置/进程快照在本实验four_x_startup_snapshot。远端run仍为 `czl@100.73.28.99:/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109`，禁止改冻结源码/请求，不动服务器仓库或其他任务。

---

> 最新用户更改（20:20）：容差轨迹动作7维损失4倍、名义1倍、容差6维组权重0.5不变；SwanLab同时记录原始/加权及分组贡献。旧“无额外4倍”描述已被覆盖。训练父3312670已确认后SIGSTOP以阻止旧协议启动GPU，归一化3312672及8workers继续，不重算。准备版本化更新，完整保留旧请求/源码及归一化凭据；不能误判父进程T为故障或直接CONT旧父。后续见four_x_protocol_update/。

# LIBERO joint13：按任务成功候选池随机替换20%容差轨迹

状态：2026-11-09，本地及远端数据完整审计通过，正式训练流程已唯一启动，正在重新计算state8/actions13归一化；GPU10000步训练与最终审计待完成。

## 协议

保留1875同源样本、312374帧、40任务及每个episode/demo身份。候选仅来自已审计的容差物理回放成功清单：必须完整执行、无求解拒绝、最终原生谓词真。逐任务对候选成功数量N取round(N×0.2)，固定随机种子42、按原demo编号排序后无放回抽样，保存确定性逐任务子种子/完整候选与入选清单。当前1503成功来源，按任务取整合计301替换、1574名义；容差占最终1875的16.05%，不是1875×20%。

入选样本：用同源容差回放的pre_action state8、256双相机MP4逐帧解码图像替换原名义观测；视频已旋转180一次，转换不再旋转，MP4有损事实保留，不能宣称与原渲染无损一致。原物理名义actions7逐帧完全一致，episode/frame/task/timestamp/语言和搜索六方向rad容差标签保持同源；trajectory_is_tolerance=1。未入选样本完整保留原名义观测/动作和搜索标注，flag=0。不替换为实际关节伺服动作，不用实际释放量充当搜索容差。

原始名义、容差回放与标签不改写。用户随后授权退役未训练的纯名义joint13转换副本；新输出未替换Parquet可硬链接复用，仅对入选301条写新独立文件，所有写入通过临时文件原子替换，不能原地写硬链接。源文件/请求SHA和抽样均须独立审计。用户已明确确认未实际修改标注，本次沿用原1851＋补24标注及对应已审计容差回放。

训练沿用既定6卡0,1,2,3,4,6，原pi05_base+官方LoRA，nostate48tokens，horizon10，batch72，8workers，FSDP2，10000步，warmup333，5e-5 cosine→5e-6，log10/save2000/keep2000，动作/容差组权重1/0.5。此次没有新增容差样本4倍动作权重。重新计算混合数据专属state8/actions13归一化，使用新run及SwanLab在线实验，完成上传及训练独立审计才宣布完成。

## 本地源数据地址

- 原名义：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101`。
- 已退役纯名义joint13（数据文件已删除，仅小型元数据/退役凭据保留）：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109`。
- 容差回放及成功索引、全量审计：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1`下`successful_sources.json`、`independent_final_audit.json`、`plan.json`；各回合`episodes/<suite>/<task>/demo_N/{main_rgb.mp4,wrist_rgb.mp4,trajectory.json,states.npz,report.json}`。
- 搜索母集：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107`；24补标：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_missing24_baseline_observation_20261109`。
- 新混合集（已组装，审计中）：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109`。
- 抽样请求/清单、转换记录、独立审计、远端run、checkpoint和SwanLab链接：待生成。

不恢复旧纯名义上传/训练监督，不修改其他任务或viewer，不重新rollout或按成功率挑选重试。

抽样已冻结：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109/replacement_request.json`；本记录`selection.json`可读清单。首个pilot为episode2，89帧，容差MP4/state替换后约10.0MB，原名义约11.1MB；动作/标签/时间戳等未变且严格Parquet往返通过。3项按任务分母/取整/可复现性/拒绝失败候选测试通过。当前磁盘余量约4.8GiB，301新文件预计约5GiB；已询问是否允许退役从未训练的纯名义转换副本，未获答复前不删除。

301条替换文件已在空间充足的临时根`/tmp/libero_joint13_successmix20_replacement_stage_20261109`以6并行转换中（主机47516，stage_launch.json和stage_conversion.log）。该临时目录只包含301条替换文件，不是完整训练数据集；本地最终路径仍以本页正式新混合集为准。原未训练纯名义副本尚未删除，待用户空间选择。

实现与验证：`/media/czl/sata/franka_my_code/openpi/examples/libero/build_successmix20_joint13.py`进行固定随机抽样及MP4/state替换；独立审计`audit_successmix20_joint13.py`将重查全部1875动作/标签/标识、301条双MP4全帧解码像素和pre_action状态、80个任务×类型实际LeRobot10步序列。pilot独立结果在本记录`pilot_independent_audit.json`，scope仅首条episode2/89帧，不代表全批次。

服务器新run已准备但未训练：`czl@100.73.28.99:/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109`。原服务器仓库SHA保护未变，独立source_overlay77文件，stage=config构造通过。新训练请求草案在本记录`training_request_draft.json`；正式冻结请求、归一化、checkpoint、SwanLab和训练审计待生成。原暂停纯名义run保留，不再启动。

样本身份按suite/task_id/原demo编号绑定，不把重构report.seed常量当作不同样本。随机种子42用于抽样，训练seed42沿用旧配置，各子任务hash派生种子记录到逐任务表。入选301来源仅取最终原生成功且完整执行的回放，不增加或重复样本，不改变帧数和时间轴。

用户已明确授权：“之前全名义轨迹的容差数据集半成品直接删除”，授权证据storage_authorization.json。服务器对应部分上传572文件/9,162,713,516字节已定向删除，元数据保存在新远端run/retired_nominal_upload_metadata，删除凭据nominal_partial_upload_deletion_verification.json；原名义数据infoSHA保持不变。

本地301替换全部生成，共47564帧、5,573,612,131字节；原1875总帧不变。当前组装会先保护未入选1574表的硬链接、归档旧转换请求/审计/全部收据、核对原1875表SHA，再删除仅旧转换的Parquet命名空间，随后将301替换文件合入新目录。没有删除原始名义或容差物理记录。

监督实现两次启动问题均保留：attempts/0001_dispatch_before_upload_state（训练调度早于上传状态创建，未触发训练）；attempts/0002_assembly_interpreter（组装以systemPython导入PyArrow失败，删除前退出；改用现成venv，无依赖安装）。确认原进程退出后恢复，同301候选/抽样请求未改动。

## 旧转换副本删除核验

本地旧转换目录已无Parquet，1875个旧路径全部清理。1574个仍需使用的表先保留到新混合集，301个被替换的旧表释放5,863,544,058字节；新301个替换表共5,573,612,131字节。原始名义数据仍为1875个Parquet。小型配置、请求、收据与审计记录保存在本记录`retired_nominal_joint13_conversion_metadata/`，删除逐文件证据为本记录`nominal_conversion_retirement.json`。该删除不表示新数据或训练审计完成。

远端旧上传目录已核实不存在，删除572文件、9,162,713,516字节，原名义数据info SHA保持一致。远端证据：`czl@100.73.28.99:/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109/nominal_partial_upload_deletion_verification.json`。旧run元数据保留、旧训练未启动。

当前运行交接见[HANDOFF.md](HANDOFF.md)，所有进程应按实际命令和锁复核，不能重复启动。上文“待授权/生成中”等段落为保留的历史过程，最新状态以本节和运行状态文件为准。

## 本地混合集完整审计通过

全1875条/312374帧动作、标识、容差标签与来源审计通过：1574条名义原图/state一致；301条容差双MP4所有帧解码像素和pre_action state对应一致；全40任务×两种来源80个实际LeRobot十步序列正确，缺标0。审计路径 `/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109/meta/joint13_independent_audit.json`，本记录 `dataset_independent_audit.json`为小型副本。审计与组装进程已真实退出，上传82812已进入全文件SHA清单生成，训练调度82814仍等待远端校验。

实际传输已启动：本地rsync120234，远端rsync3219491/3219492；新目录已开始提交Parquet，当前是部分上传，不是远端验收完成。传输清单1890文件/38,388,719,043字节（另有transfer_manifest本身），manifest SHA256 `c83584914ad3eb169dcee7bb489990f562a2db4175d8d2ebd121f1002aca5d8d`。上传监督和训练调度仍活跃；训练启动须远端逐SHA/原动作核对/实际读取及磁盘预算全部通过。

上传间隔只读核验入口为本记录 `observe_progress.py`，不启动/重启任务；最新机器可读观测在 `verified_wait_upload.json`。本轮18:18–18:24的8次探测：提交表197→293、最新4.18GB，所有本轮监督真实命令匹配且无报错，训练尚未启动。

## 传输方式优化（数据与训练请求不变）

原rsync真实活跃，网络吞吐较低；独立检查证明服务器Parquet重写、PNG编码，以及名义/容差/Long三条97/89/307帧重建均与本地冻结SHA完全一致。将主动切换到240,203,401字节的紧凑包（数值IPC、原MP4和schema），服务器只提交与原冻结SHA完全一致的Parquet，已有正确文件直接保留。此为传输方式变更，不是将旧上传误判死亡或重测数据。单文件重建若不一致则保留证据并传原Parquet，最终仍运行原远端全量SHA/原动作与实际LeRobot验证，之后才训练。

证据及计划：本记录`transport_transition/plan.json`、`transport_transition/three_case_compatibility.json`；紧凑请求 `/tmp/libero_successmix20_exact_transport_20261109/transport_request.json`；构建入口本记录`prepare_exact_transport.py`，新上传入口`upload_exact_transport.py`，服务器独立重建入口 `/media/czl/sata/franka_my_code/openpi/examples/libero/reconstruct_joint13_exact_transport_server.py`。现有训练调度和服务器完成监督保留；只切换本次旧上传父子进程，不涉及原始数据/其他用户进程。

传输切换已实际执行：旧本地上传82812及rsync120234/120235退出，旧远端3219491/3219492退出；本次原训练调度82814、SSH主连接和远端最终监督3210109保留。新上传136691、rsync136694真实运行，803个现有Parquet保留。切换凭据 `transport_transition/before_stop.json`、`local_stop_verification.json`，新日志 `upload_exact_transport.log`。紧凑传输不是另一次数据生成实验，原manifest SHA保持c83584914ad3eb169dcee7bb489990f562a2db4175d8d2ebd121f1002aca5d8d。

紧凑传输完成，共240,203,401字节；服务器原SHA重建已全1875通过：802已有正确文件复用、1073重建文件提交、错误0、无需原文件回退。源文件与本地冻结Parquet逐字节一致；重建真实进程3267870已退出，现由原verify_successmix20_joint13_upload_server.py独立检查全部1890文件及原动作/实际读取。详细逐文件凭据在本记录exact_transport_reconstruction_completion.json，小结exact_transport_reconstruction_summary.json。

## 正式入口与归一化已启动

原独立远端验证全1890文件SHA、全1875原动作/标识、1574未替换观测、实际LeRobot读取均通过，缺标0；本地凭据lerobot_upload_verification.json。训练调度82814已完成唯一启动并真实退出，上传136691也正常退出。服务器总控3312670、归一化3312672活跃，8个真实worker已核对；启动快照server_startup_snapshot/与training_startup_verification.json记录了实际配置逐字段与请求一致。冻结请求training_request.json SHA9077cac73ceae17e0e9226af65d5221cd1b0089b6a681ba978bee22a6bcb9917，模型原pi05_base/LoRA、batch72、horizon10/有效13/接口32、组损失1/0.5、FSDP2、10000步、warmup333与cosine等全部匹配。当前处于统计阶段，未将此计为GPU训练完成。

只读目标缩放检查tolerance_target_scale_check.json覆盖312374帧，六方向非零比例约19.10%–57.75%，q99均为30°，全部有限；该检查不替代本次服务器4338batch完整归一化。磁盘预算training_disk_budget.json已通过，按缓存＋五份checkpoint及8GiB余量核算。归一化完成、实际6卡训练、SwanLab在线新实验、10000步及五checkpoint完整CPU恢复审计仍待后续验证。


20:27：唯一4x切换监督3334760已在服务器实际运行，stage=waiting_for_original_normalization。仅原总控3312670有意SIGSTOP，统计3312672继续、当前无GPU训练step。监督将确认原4338batch日志/统计维度有限/真实exit0后，写原统计凭据，定向结束旧总控与旧结束监督3210109，完整保存旧请求/源码到run/four_x_protocol_update/before_4x，部署新4x源码并做CPU真实LeRobot/归一化/Observation/JIT损失预检，通过才唯一启动训练及新最终审计监督。若transition_state=failed应查看原因；不能CONT旧父或重复启动。归一化不因损失权重变更重算，保留原receiptSHA，并独立证明旧输入与新输入去掉仅用于loss的weight字段逐项一致。

新版参数：仅301容差观测来源的动作7维loss×4；1574名义×1；容差标签6维组权重0.5不变。总loss=mean(w_i*action_mse7+0.5*tolerance_mse6)，不按权重和重新归一化。SwanLab在线记录loss/action、loss/rotation_tolerance、loss_weighted/action、loss_weighted/rotation_tolerance、loss/total、loss/action_nominal_contribution、loss/action_reweighted_contribution、batch/action_loss_weight_mean、batch/reweighted_action_fraction；每10步记录，最终云端逐行复核4x贡献恒等式。34测试通过含JIT真实梯度4倍与容差不变/无填充梯度。用户未修改数据、标签、模型或其余训练超参数。

新证据本地绝对根：/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/four_x_protocol_update；实际远端四倍配置/新进程/预检待统计结束后生成，此前training_request.json与startup_verification仍代表旧归一化阶段，不能把它们说成新GPU已启动。远端状态/run/four_x_protocol_update/transition_state.json（完整run路径见服务器章节），完成后应取回normalization_reuse_verification.json、training_preflight_verification.json、transition_completion.json及新的training_request/actual_config/launch。


## 首条loss0.0958与raw action0.4343的口径诊断

只读核对两次实际配置及首10步均值：旧action_loss_groups=null，对32维（有效动作7+填充25）的平方误差求均值；本次先对动作7与容差6分别求均值，再仅对容差观测来源动作组施加4倍权重、容差组施加0.5。新raw action0.4343未含4倍。相同动作7误差贡献从分母32改为7会放大32/7=4.5714；将当前7维贡献换算回分母32为0.095003，接近旧总loss0.0958。旧总数还含25填充维预测残差，因此此为贡献换算，不是同批次误差精确对照。

实际读取两份norm_stats：动作前7维mean/std/q01/q99全部逐值相等（最大差0），排除动作归一化尺度变化。当前weighted action0.6414、weighted tolerance0.4236，总1.0650；模型/数据/早期更新也有差异，不能用不同口径数值认定动作误差变差4倍或用于性能排名。两次首条均为Step10所记录前10步均值，不是孤立第1步。未启动额外实验、未改动当前训练。证据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/loss_scale_comparison.json`；原始两份远端训练run与本地源码绝对路径均在该JSON中。

补充口径：前7维动作的逐维flow-matching速度MSE定义没有改变，固定同一逐维误差时仅均值系数从1/32改为1/7，再施加来源样本权重1/4；这些系数参与反向传播。整体目标还新增6维容差监督、去除剩余19填充维监督，并混入301条容差观测，因此训练不整体等价。实际base checkpoint元数据确认输入投影32×1024、输出1024×32，主干与原32维动作头未因joint13重新随机初始化；base无LoRA项，新增官方LoRA照常初始化，双方seed42。未保存历史step0数组，不能把相同协议冒称逐参数初值已比对。详见loss_scale_comparison.json。
