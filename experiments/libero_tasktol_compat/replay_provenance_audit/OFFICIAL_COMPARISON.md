# 官方 RLDS 与本地重录对照 · 2026-10-20

本地旧环境重录在本次样本中接近官方发布版；官方重生成数据相对原始记录同样存在厘米级轨迹差。相对原始记录的偏差及约 15% 未保留比例，不能单独证明本机 LIBERO 环境异常。此结论不涵盖 MuJoCo 3.10 的迁移结果。

官方来源：[openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)，固定提交 `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`。

## 完整数据集数量

| 任务集 | 官方 train episode | 本地成功重录 |
|---|---:|---:|
| libero_spatial | 432 | 436 |
| libero_object | 454 | 452 |
| libero_goal | 428 | 427 |
| libero_10 | 379 | 384 |
| 合计 | 1693 | 1699 |

官方数量来自四份 dataset_info.json 的 train/shardLengths 求和，本地数量来自完整重录 ledger。以四套各 500 条原始示范计，官方未保留 307 条（15.35%）；本地重录失败 301 条（15.05%）。官方发布元数据未逐条说明未保留原因，不能把 307 条全部认定为同一种失败。

## 直接轨迹对照

每套取首个 shard 的前 4 个完整记录，共 16 条；16 条均通过完整 float32 动作序列精确匹配到唯一本地示范，涵盖 2759 帧。TFRecord 长度与内容 CRC32C 均校验通过。此为便利样本，并非随机抽样或全量一致性检验。

官方与本地均按同一保留动作的动作前观测比较。对原始记录的比较另作动作前／后错位处理，沿用 audit.py 已核实的时序；仅统计末端位置的欧氏距离，不以 JPEG 像素误差衡量动力学。

| 任务集 | 匹配条数 | 本地对官方：最大末端差 | 官方对原始：最大末端差 |
|---|---:|---:|---:|
| libero_spatial | 4 | 0.899324 mm | 18.929032 mm |
| libero_object | 4 | 0.001038 mm | 11.812867 mm |
| libero_goal | 4 | 0.000060 mm | 17.166161 mm |
| libero_10 | 4 | 1.277936 mm | 83.958966 mm |

15/16 条的本地对官方最大末端差小于 1 mm，全部样本最差 1.277936 mm。

本地与官方不完全相同：保留集合数量已有差异，个别关节、夹爪状态也有差别。末端接近不证明物体状态、接触力或所有关节逐帧等价；本次未重新运行仿真，未比较成功判据实现。

## 逐条结果与留存图像

| 任务集 / demo | 本地对官方末端最大差 mm | 官方对原始末端最大差 mm | 主视角对照 |
|---|---:|---:|---|
| libero_spatial / demo_40 | 0.123347 | 6.918408 | [图片](libero_spatial_00_demo_40.png) |
| libero_spatial / demo_14 | 0.899324 | 11.352300 | [图片](libero_spatial_01_demo_14.png) |
| libero_spatial / demo_42 | 0.495155 | 5.670882 | [图片](libero_spatial_02_demo_42.png) |
| libero_spatial / demo_5 | 0.001531 | 18.929032 | [图片](libero_spatial_03_demo_5.png) |
| libero_object / demo_31 | 0.000126 | 8.460530 | [图片](libero_object_00_demo_31.png) |
| libero_object / demo_11 | 0.000017 | 7.516442 | [图片](libero_object_01_demo_11.png) |
| libero_object / demo_21 | 0.001038 | 11.812867 | [图片](libero_object_02_demo_21.png) |
| libero_object / demo_38 | 0.000434 | 4.616216 | [图片](libero_object_03_demo_38.png) |
| libero_goal / demo_10 | 0.000059 | 5.070019 | [图片](libero_goal_00_demo_10.png) |
| libero_goal / demo_13 | 0.000060 | 10.788676 | [图片](libero_goal_01_demo_13.png) |
| libero_goal / demo_6 | 0.000060 | 7.162102 | [图片](libero_goal_02_demo_6.png) |
| libero_goal / demo_17 | 0.000059 | 17.166161 | [图片](libero_goal_03_demo_17.png) |
| libero_10 / demo_19 | 0.000030 | 6.804866 | [图片](libero_10_00_demo_19.png) |
| libero_10 / demo_16 | 0.955279 | 11.511037 | [图片](libero_10_01_demo_16.png) |
| libero_10 / demo_39 | 1.277936 | 83.958966 | [图片](libero_10_02_demo_39.png) |
| libero_10 / demo_10 | 0.000061 | 22.030724 | [图片](libero_10_03_demo_10.png) |

复现：在 openpi 根目录运行 `.venv/bin/python experiments/libero_tasktol_compat/replay_provenance_audit/official_rlds.py --offline`。

`official_comparison.json` 保存逐条任务名、匹配 demo、末端／关节／夹爪差异；`official_rlds_samples/` 保存固定提交的元数据、完整 TFRecord 前缀、CRC 校验与 SHA256。仅导出和查看主视角图片。原始数据、处理后 HDF5 和冻结 mask 均未修改。
