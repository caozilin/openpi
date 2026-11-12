> 用户最新明确要求暂停：将重新处理部分容差标注。上传/调度/远端结束审计监督均已停止，训练未启动。不要继续下面旧运行安排；恢复须获用户明确指令，先核对新标签。见user_pause.json及README最新章节。

# 当前授权目标与运行交接（2026-11-09）

目标仍 active：生成同款全部1875名义LeRobot并加搜索容差标签/轨迹类型，上传后用同样6卡参数训练10000步（有效13维→原32维接口），完成独立训练审计。不能以补标或转换结束作为整个目标完成。

用户对24条缺标的最新指令：“基线并不要求一定可行。直接补标尝试”。已按本文README中明确的基线观测协议单独补标24/24、660候选审计通过；原5578份保护文件SHA不变。现在1851+24全部标签有效，不屏蔽、不删除样本。不得重跑1851或使用容差物理回放数据训练。

## 当前真实进程

- 转换主机21170：1875全部完成，已退出；数据根 `/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109`，`conversion_completion.json`为转换完成，尚不等于独立审计完成。
- 本地监督21479：`finish_local_dataset.py`，自动等待转换退出并启动全量独立审计。
- 数据审计23984：`examples/libero/audit_nominal_joint13_dataset.py`；最近1050/1875逐字节/标签通过。完成后40任务实际LeRobot读取（每任务首中末+10步序列），缓存仅在`/tmp/libero_joint13_audit_hf_20261109`，不能在剩余4.9GiB的数据盘生成完整Arrow缓存。
- 上传监督22163：`upload_joint_dataset.py`，仅当本地全量独立审计通过才建立transfer_manifest并rsync，使用远端旧名义数据作为`--copy-dest`独立复制差分基底，不是硬链接。上传后自动调用远端`verify_joint13_upload_server.py`验证全部SHA、1875原列逐字节相等、4suite实际LeRobot读取。当前等本地审计，尚未上传。
- 训练调度24676：`start_joint_training_when_verified.py`，仅当上述远端校验通过、磁盘预算通过才冻结最终training_request并启动一次服务器pipeline；当前等待校验，训练尚未启动。首次自动审批仅超时，按工具准许重试一次后成功启动，未被判不安全。
- 进程以对应launch/process JSON、真实主机`/proc`命令和锁为准；沙箱PID不可用于判断。活跃不重复启动。任何failed先查日志，保留已完成结果。

本记录中的`local_pipeline_state.json`、`upload_state.json`、`training_dispatch_state.json`和各log为当前状态。以上监督为主机持久进程，不需重启。原24补标run总控19405和全部搜索/审计已退出锁释放。

## 服务器

专用SSH socket `/tmp/libero_joint13_20261109.sock`，ControlPersist8h；服务器`czl@100.73.28.99`。已有登录授权，仅交互使用凭据，严禁写密码到文件或日志。目标GPU0,1,2,3,4,6；GPU5桌面和GPU6AnyDesk不动。最新目标GPU0%且<100MiB。

新run：`/mnt/SharedData/czl/runs/pi05_libero_nominal_joint13_lora_6gpu_bs72_10k_20261109`。Python `/home/czl/openpi/.venv/bin/python`。新数据：`/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109`（待上传）。

已从服务器原仓库只读复制77份Python到新run/source_overlay，再加入3份新LIBEROjoint13模块；原仓库保护SHA齐全且未改动，未安装依赖。原服务器源码、环境和其他任务不得修改。`overlay_manifest.json`SHA eeb7fd1a93daa668a41c46454d44e202cad88b967283e8b6a4b5f7b3913bd758；entry位于overlay之外，有单独脚本SHA。

服务器stage=config构造检查通过。entry之后仅作启动前JSON分组→immutable tuple规范化，`server_entry_prelaunch_amendment.json`记前后SHA。最终脚本SHA以本地training_request_draft/服务器实际脚本为准，调度器会核对。服务器现有training_request.json仍是未冻结草案，调度器在训练未启动前会保存旧草案并写最终请求，不得拿草案声称训练开始。

磁盘SharedData上传前约137GiB，旧单checkpoint9,567,313,920字节量级；调度器实际测量上传后free、数据Parquet bytes与旧checkpoint大小，预算=1.03×新Parquet作为Arrow缓存+5×1.05×旧checkpoint+8GiB余量，不足即失败保留并报告，不自动删除旧文件。训练与归一化使用SharedData，不在home根盘另建完整缓存。

## 精确训练协议

原pi05_base重新初始化（不是上次10000步模型），Gemma2B与300Mexpert官方LoRA/bfloat16；nostate48tokens、horizon10、action_dim32预训练投影保留。数据有效动作13=原physical7+6signedrad，先13归一化再pad32；损失仅action0:7权重1和rotation_tolerance7:13权重0.5；padding不计分组损失。全1875=312374帧，trajectory_is_tolerance全部0。保留原256双相机PNG字节/state8/actions7，无OSC或额外delta。

seed42、batch72=6×12、8workers、FSDP2、10000步、warmup333、5e-5 cosine→5e-6、AdamW(.9,.95,eps1e-8,wd1e-10,clip1)、EMA None、log10/save2000/keep2000。SwanLab必须开启，cfg online、SDK env cloud，run/swanlog。norm重新计算state8/actions13，不得复用旧actions7统计。

checkpoint父目录预计：`/mnt/SharedData/czl/checkpoints/pi05_libero_nominal_joint13_nostate_6gpu_bs72/Pi05_libero_nominal_joint13_NoState_LoRA_6gpu_bs72_10k_20261109`。

## 仍需完成

监测本地审计→上传/远端审计→norm→真实6卡train；失败先诊断不可盲目重复启动。数据审计不能跳过。训练完成需：真实10000条步数、1000条10..10000有限日志、exit0且总控/train退出/锁释放，五份2000/4000/6000/8000/10000 checkpoint CPU完整恢复参数与train_state、有限性与精确step及norm SHA，SwanLab云端真实指标和FINISHED，独立核对实际配置/有效13维/名义来源；不能只看目录和completion。最终训练审计尚未实现/执行，必须继续完成。

更新本记录README/RESULTS、来源索引与最终地址。全部目标达到才update_goal complete；本目标无token预算，无需token报告。不扩展额外训练、benchmark、容差物理回放或标注搜索，不停viewer，不删除原始数据，不关机、不归档。
