# joint13预测容差：40任务×seed0–9

完整400回合成功366/400（91.50%），全部独立审计通过。

|套件|成功/100|
|---|---:|
|libero_spatial|92/100|
|libero_object|95/100|
|libero_goal|92/100|
|libero_10|87/100|

|套件|task_id|成功/10|
|---|---:|---:|
|libero_spatial|00|10/10|
|libero_spatial|01|9/10|
|libero_spatial|02|10/10|
|libero_spatial|03|10/10|
|libero_spatial|04|8/10|
|libero_spatial|05|10/10|
|libero_spatial|06|9/10|
|libero_spatial|07|7/10|
|libero_spatial|08|10/10|
|libero_spatial|09|9/10|
|libero_object|00|9/10|
|libero_object|01|10/10|
|libero_object|02|10/10|
|libero_object|03|9/10|
|libero_object|04|8/10|
|libero_object|05|10/10|
|libero_object|06|10/10|
|libero_object|07|10/10|
|libero_object|08|10/10|
|libero_object|09|9/10|
|libero_goal|00|9/10|
|libero_goal|01|10/10|
|libero_goal|02|9/10|
|libero_goal|03|9/10|
|libero_goal|04|10/10|
|libero_goal|05|8/10|
|libero_goal|06|9/10|
|libero_goal|07|10/10|
|libero_goal|08|10/10|
|libero_goal|09|8/10|
|libero_10|00|8/10|
|libero_10|01|10/10|
|libero_10|02|9/10|
|libero_10|03|10/10|
|libero_10|04|10/10|
|libero_10|05|10/10|
|libero_10|06|6/10|
|libero_10|07|10/10|
|libero_10|08|8/10|
|libero_10|09|6/10|

历史中断：6条未提交回合server keepalive超时；66条已提交成败SHA均保留，仅未完成条目续跑。传输增加同packet同seed同模型重连，未改变物理协议。
模型为successmix20 joint13最终10000步；使用预测有向容差，0–30°裁剪，保留共享阶段和heading保护，不使用手工任务mask。连续失败20帧保持参考并终止，首次原生成功后继续20步；最终谓词单独记录。当前共享运行时与历史20261025快照不同，结果不声称为仅模型变化的严格因果对照。

原始本地地址：/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110
请求：/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110/evaluation_request.json
审计：/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110/evaluation_independent_final_audit.json
