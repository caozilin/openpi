# joint13预测容差400回合交接

最新用户只测评最终10000步模型；40任务×官方seed0–9，共400；新模型预测6方向容差。原训练及全部独立审计已通过，不重训、不测其他checkpoint。

本地run：/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110
监督：/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/pipeline.py，真实PID19496；总控19521；六客户端[19528, 19530, 19531, 19532, 19533, 19534]。以真实主机/proc命令/start_ticks和pipeline/evaluation锁为准，活跃不重复启动。当前stage=evaluating，首6真实成功并只读审计通过，后续尚未完成；summary和真实result计数为准。

先读AGENTS、docs/TRAINING_SERVER.md及本README。SSH /tmp/libero_joint13_eval400_20261110.sock；远端 /mnt/SharedData/czl/runs/libero_eval_joint13_successmix20_predtol_400_20261110，manager3914268，6副本3914269–3914274，GPU0/1/2/3/4/6，只推理。六18760–18765本地-L、loopback服务。凭据仅在内存从用户授权私有文件读取。别动GPU5/其他用户任务、viewer、旧训练socket。

冻结请求SHA604ae8feda0cb6efd441f0c2ab8a7243ab7e000e8180554c30485ae12457f004；479来源SHA。当前shared runtime在run/frozen_shared_runtime；IPOPT20、EMA10、20Hz/500Hz、replan5/predict10，动作7物理增量、容差6rad按rx-/rx+/ry-/ry+/rz-/rz+逐帧同chunk配对，裁剪0..pi/6，mask由非零轴派生，保留shared因果phase/ENG-DIS/heading保护，无手工task表。求解失败保持上一接受参考20帧终止，首次成功后继续20步。不得改控制/预算或按成败选择重测。

历史本地监督17441在首6审计实现错误后停止全部本地父子；它们已真实退出，原6*4 artifact SHA保护，未提交目录归档，服务器副本未重启。冻结原审计不变；只读修正audit_evaluation.py＋audit_bound_replay.py及SHA在audit_implementation_amendment.json，canonical重放EMA滤波和收缩恢复硬盒，不执行求解/物理/推理。初版错误是以原始预测盒代替EMA后的声明/有效硬盒。新初审已通过6，request/模型/物理协议不变；不要重新运行旧固定盒审计。

监督已改用上述修正审计。完成400且6clientexit0才全量只读审计，核验动作/状态/失败保持/25physics/实际滤波边界/恢复硬盒、官方初态、原生首末成功谓词和全视频。通过后生成predicted_tolerance400_report.json、结果CSV、PREDICTED_TOLERANCE_400_REPORT.md和本实验RESULTS.md，定向退出本manager及6children、取消本6-L及专用socket，保留viewer/全部源数据。不能凭summary宣布完成。最终需evaluation_completion/audit/task_completion/pipeline_state complete和真实退出、锁释放、定向清理凭据。异常先检查pipeline.log/状态/客户端/真实进程，不盲目重复启动。所有成功与失败已提交artifact不得覆盖，基础设施错误单列。

用户最新汇报格式：每15分钟必须逐行列Spatial/Object/Goal/Long，显示成功数/已完成数、成功率及该套件N/100进度；未开始写未开始，总体只能补充。heartbeat libero-400-15已更新，不新建重复监督。

最新17:02同参数恢复：监督30768，总控30823，六本地wrapper30824..30829；之前19496/19521等全部退出。66已提交成败SHA保护，6个未提交server keepalive超时归档attempts/0003_server_keepalive_ping_timeout。新传输wrapper run_client.py＋transport_reconnect.py、最多3次同packet同seed同metadata重连，原600秒总deadline，6单测通过，SHA见transport_recovery_amendment.json。冻结物理入口/请求/模型/控制不变，原66不得重测。远端manager3914268及6副本未重启，后续复用活跃实例。不要运行旧无重连入口。历史6次infra单列，当前错误从活动episodes中计数。完整审计用既有修正只读checker，不用旧固定盒审计。
