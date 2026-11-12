# 长期训练服务器使用说明

本文件是本地openpi项目的长期服务器说明。**任何Agent登录、上传、训练、推理、清理文件或管理远端进程前，必须先阅读本文件及相关实验记录。** 具体run的冻结请求与已授权用卡优先于本文的历史快照。核实日期：2026-11-09；动态占用/余量每次重新查询。

## SSH、密钥与本机口令

| 项目 | 当前信息 |
|---|---|
| 服务器 | `100.73.28.99`，Tailscale网络，hostname=`server` |
| 用户 / 端口 | `czl` / `22` |
| 直接连接 | `ssh czl@100.73.28.99` |
| 本机私有凭据文件 | `/media/czl/sata/franka_my_code/openpi/.local/server_access/training_server.json` |
| 本机私钥路径 | `/home/czl/.ssh/id_ed25519`；仅引用，不能复制或打印内容 |
| 本机公钥路径 | `/home/czl/.ssh/id_ed25519.pub` |
| 本机公钥指纹 | `SHA256:XDVvzqZGVBnDQYeFEll+jO19XhDaC4YNW8yJzgLgfu0`（ED25519） |
| 服务器ED25519主机指纹 | `SHA256:nUDsh1J/rQKcfPDEvBngoLWtFG+gbpoodOgNNyUTsag` |
| 服务器ECDSA主机指纹 | `SHA256:u98xAGvTUFz/CbZUS9QDjvLRPx5DhYnNsdVpAqGDRdU` |

用户明确授权将口令保存在上述**本机私有文件**，目录权限700、文件600，并同时由`.gitignore`和本机`.git/info/exclude`忽略。AGENTS、本文、实验文档、日志和源数据清单不放口令/私钥内容。需要时仅在进程内读取私有JSON并交互输入；不要cat/echo、写入命令参数、环境转储或日志。不得`git add -f`该目录，也不得通过rsync、打包、训练payload、分享或上传带出该文件。

2026-11-09独立禁用ControlMaster复用、显式指定本机ed25519密钥的BatchMode测试被服务器拒绝；**本机有密钥不等于该服务器已接受它**。当前可使用既有SSH控制连接；连接失效时，已有用户授权允许从本机私有文件读取口令重建交互连接，无须再次索要同一口令。本文没有授权安装/替换远端authorized_keys、生成新密钥、修改SSH/Tailscale/防火墙配置或放开公网入口。

现有known_hosts在`/home/czl/.ssh/known_hosts`；保留主机验证，遇到指纹变化先核实，不使用`StrictHostKeyChecking=no`绕过。

每个任务使用独立控制socket。示例（根据run选择唯一文件名；不要占用其他任务socket）：

```bash
TRAIN_SSH_SOCKET=/tmp/openpi-training-example.sock
ssh -M -S "$TRAIN_SSH_SOCKET" -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -fN czl@100.73.28.99
ssh -S "$TRAIN_SSH_SOCKET" -O check czl@100.73.28.99
ssh -S "$TRAIN_SSH_SOCKET" -o BatchMode=yes czl@100.73.28.99 'hostname'
```

Socket失效只说明连接需恢复，不说明远端任务退出。连接恢复后先查真实PID/命令/start_ticks、锁和日志，不能重复启动。收尾只取消本任务转发/控制连接，不能关闭别的任务连接。

## 项目、环境与大文件位置

下表均为**服务器地址**，不能作为本机文件路径使用。

| 内容 | 服务器路径 |
|---|---|
| 项目仓库 | `czl@100.73.28.99:/home/czl/openpi` |
| 已有Python环境 | `czl@100.73.28.99:/home/czl/openpi/.venv/bin/python` |
| 本账号大文件根 | `czl@100.73.28.99:/mnt/SharedData/czl` |
| LeRobot训练数据 | `czl@100.73.28.99:/mnt/SharedData/czl/lerobot/<owner>/<dataset>` |
| 数据归一化/模型assets | `czl@100.73.28.99:/mnt/SharedData/czl/assets/<config>/<repo_id>` |
| checkpoint | `czl@100.73.28.99:/mnt/SharedData/czl/checkpoints/<config>/<experiment>` |
| 独立run、请求、overlay、日志、SwanLab和审计 | `czl@100.73.28.99:/mnt/SharedData/czl/runs/<run>` |
| HF数据Arrow缓存 | `czl@100.73.28.99:/mnt/SharedData/czl/hf_datasets/<owner>/<dataset>` |
| 已有pi05_base缓存（历史例外，复用） | `czl@100.73.28.99:/home/czl/.cache/openpi/openpi-assets/checkpoints/pi05_base/params` |

`/mnt/SharedData`已核实为服务器`/dev/sdc1`的ext4本地磁盘，不是本机共享挂载。原有base缓存按既有路径复用，不自动移动或重复下载。新增大数据/缓存/checkpoint明确配置到本账号`/mnt/SharedData/czl`命名空间，不把大文件塞入仓库，不写他人的目录。

复用现成环境，不自行升级/重装共享依赖，不修改全局shell/conda/CUDA配置。当前服务器仓库没有AGENTS.md；本地AGENTS是本工作流的必读入口。远端代码修改应优先采用run专用source_overlay/独立脚本，只上传明确manifest列出的代码；保持原仓库和其他任务环境不变。若以后用户明确要求在服务器直接开Agent，应先提供同等规则入口；本文本次仅落在本地项目。

2026-11-09空间快照：SharedData与/home都只剩约65GiB，分别约97%/99%已用。这是历史快照，不是可分配预算。当前run已为完整Arrow缓存、五checkpoint和8GiB余量核算预算；不能将尚未写出的checkpoint预留空间当作新任务空闲空间。新任务启动前重新查`df`，计算数据＋缓存＋保存策略＋其他运行中任务预算，空间不足先报告，不自动删模型/缓存/他人数据腾空间。

## 显卡与多人共用规则

已核实硬件为**7张NVIDIA GeForce RTX 3090，每张24576MiB（24GiB）**。GPU索引与进程占用会变化，不把历史分配当作永久独占权。

- 本次用户已授权的六卡训练使用`CUDA_VISIBLE_DEVICES=0,1,2,3,4,6`。本次实际batch72=每卡12；后续任务应按各自授权和run请求决定，不能照抄后假定仍空闲。
- 2026-11-09核对：GPU5有他人的图形/办公进程，GPU6也有他人的少量桌面进程。**利用率低或仅占几百MiB不等于可以终止/抢占。** 本次训练共享GPU6，但不终止桌面进程；GPU5不用于本次训练。
- 每次启动前检查全卡显存、利用率、进程所有者、真实命令、运行目录和已有实验。进程属于czl也不代表属于当前任务；仍需run路径、PID、start_ticks及父子关系核对。
- 禁止泛化`pkill python`、`killall`、按用户名杀进程、GPU reset、重启/关机服务器，或停止他人桌面/远控服务。只在有任务授权且确认精确归属时定向管理本任务总控与其子进程。
- 长任务采用独立后台session，stdout/stderr写本run日志，记录PID、启动时间/start_ticks、命令、锁、GPU和配置SHA；再次执行前复用活跃实例。不能因一次超时、日志暂时不变或沙箱/proc不可见就判断死亡重启。
- 只清理由用户明确授权、已经核验来源且不再被活跃任务使用的具体路径。不得扫全盘、按“旧”目录名批量删除，亦不改写冻结数据/SHA掩盖问题。

常用只读检查（通过既有SSH连接执行）：

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
ps -p PID -o pid,ppid,pgid,sid,tty,stat,lstart,args
df -h /mnt/SharedData/czl /home/czl/openpi
```

## 当前实验与完成验收

长期服务器规则写在本文件；单次实验的PID、socket、模型、数据、checkpoint、日志与审计放`experiment_records/`，不能依赖长期文档中的过时占用快照。

当前joint13 successmix20训练记录：[实验总结](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/README.md)；[当前交接](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/HANDOFF.md)。本次socket为`/tmp/libero_joint13_20261109.sock`，训练总控3339995/训练3339998/完成监督3339996；它们是本次历史身份，重连时必须实际核对，不盲信PID数字。

SwanLab按具体run开启在线并核对真实云端实验与指标；不能把本地目录/链接当作已同步。训练完成需实际目标步数、exit0/进程退出/锁释放、可恢复checkpoint、有限参数与统计、完整日志及独立审计，不能凭目录名或state文件宣布完成。

数据在服务器且后台session/日志均独立时，本机关机或断开SSH不会停止服务器训练、保存、SwanLab和服务器最终审计；本地Codex的实时监督会中断。恢复本机后先读本文件与实验HANDOFF再续查。
