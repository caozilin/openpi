# joint13无容差：40任务×seed0–9

完整400回合成功368/400（92.00%），全部独立审计通过。

|套件|成功/100|
|---|---:|
|libero_spatial|96/100|
|libero_object|97/100|
|libero_goal|88/100|
|libero_10|87/100|

|套件|task_id|成功/10|
|---|---:|---:|
|libero_spatial|00|10/10|
|libero_spatial|01|9/10|
|libero_spatial|02|10/10|
|libero_spatial|03|10/10|
|libero_spatial|04|9/10|
|libero_spatial|05|9/10|
|libero_spatial|06|10/10|
|libero_spatial|07|10/10|
|libero_spatial|08|10/10|
|libero_spatial|09|9/10|
|libero_object|00|9/10|
|libero_object|01|10/10|
|libero_object|02|9/10|
|libero_object|03|9/10|
|libero_object|04|10/10|
|libero_object|05|10/10|
|libero_object|06|10/10|
|libero_object|07|10/10|
|libero_object|08|10/10|
|libero_object|09|10/10|
|libero_goal|00|9/10|
|libero_goal|01|10/10|
|libero_goal|02|9/10|
|libero_goal|03|9/10|
|libero_goal|04|10/10|
|libero_goal|05|6/10|
|libero_goal|06|8/10|
|libero_goal|07|10/10|
|libero_goal|08|9/10|
|libero_goal|09|8/10|
|libero_10|00|8/10|
|libero_10|01|10/10|
|libero_10|02|10/10|
|libero_10|03|10/10|
|libero_10|04|8/10|
|libero_10|05|9/10|
|libero_10|06|7/10|
|libero_10|07|10/10|
|libero_10|08|9/10|
|libero_10|09|6/10|

本组沿用已验证的同packet同seed同metadata有界重连wrapper，原600秒总deadline；历史容差组的6次中断不计入本组。
模型为successmix20 joint13最终10000步；客户端采用bounds_rad=None的严格无容差分支；模型预测容差仅诊断保存，不进入控制器。连续失败20帧保持参考并终止，首次原生成功后继续20步；最终谓词单独记录。当前共享运行时与历史20261025快照不同，结果不声称为仅模型变化的严格因果对照。

原始本地地址：/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110
请求：/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/evaluation_request.json
审计：/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/evaluation_independent_final_audit.json


完整终审采用audit_evaluation_v3.py：原v2审计对无容差位置/姿态残差误用L2，现按原冻结IPOPT max(abs(component))逐项复核，不改变任何阈值或回合；首块跨副本数值差不作为等值验收，见numeric_comparison_audit_amendment.json及residual_norm_audit_amendment.json。400回合原始artifact SHA未变，最终凭据final_delivery_verification.json。
