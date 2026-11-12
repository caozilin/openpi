# 40任务物理回放成功率

对应2026-11-07最新标注容差组。任务ID从0开始；每任务分母是该任务有有效标签且实际回放的全部来源，保留成功与失败，不含24条无标签来源，不含重测。成功要求完整执行且最终原生任务谓词成立；1851条已通过独立审计，1503条成功（81.20%）。

|套件|task ID|任务原文|成功/回放数|成功率|求解拒绝（包含在失败中）|
|---|---:|---|---:|---:|---:|
|libero_spatial|00|pick up the black bowl between the plate and the ramekin and place it on the plate|42/47|89.36%|0|
|libero_spatial|01|pick up the black bowl next to the ramekin and place it on the plate|40/47|85.11%|0|
|libero_spatial|02|pick up the black bowl from table center and place it on the plate|38/47|80.85%|0|
|libero_spatial|03|pick up the black bowl on the cookie box and place it on the plate|36/45|80.00%|0|
|libero_spatial|04|pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate|30/34|88.24%|0|
|libero_spatial|05|pick up the black bowl on the ramekin and place it on the plate|42/48|87.50%|0|
|libero_spatial|06|pick up the black bowl next to the cookie box and place it on the plate|45/47|95.74%|0|
|libero_spatial|07|pick up the black bowl on the stove and place it on the plate|26/43|60.47%|0|
|libero_spatial|08|pick up the black bowl next to the plate and place it on the plate|44/49|89.80%|0|
|libero_spatial|09|pick up the black bowl on the wooden cabinet and place it on the plate|35/48|72.92%|0|
|libero_object|00|pick up the alphabet soup and place it in the basket|28/50|56.00%|0|
|libero_object|01|pick up the cream cheese and place it in the basket|49/49|100.00%|0|
|libero_object|02|pick up the salad dressing and place it in the basket|41/49|83.67%|0|
|libero_object|03|pick up the bbq sauce and place it in the basket|40/49|81.63%|0|
|libero_object|04|pick up the ketchup and place it in the basket|33/48|68.75%|0|
|libero_object|05|pick up the tomato sauce and place it in the basket|44/49|89.80%|0|
|libero_object|06|pick up the butter and place it in the basket|48/49|97.96%|0|
|libero_object|07|pick up the milk and place it in the basket|43/50|86.00%|0|
|libero_object|08|pick up the chocolate pudding and place it in the basket|48/50|96.00%|0|
|libero_object|09|pick up the orange juice and place it in the basket|38/50|76.00%|0|
|libero_goal|00|open the middle drawer of the cabinet|46/46|100.00%|0|
|libero_goal|01|put the bowl on the stove|48/50|96.00%|0|
|libero_goal|02|put the wine bottle on top of the cabinet|35/37|94.59%|0|
|libero_goal|03|open the top drawer and put the bowl inside|36/45|80.00%|0|
|libero_goal|04|put the bowl on top of the cabinet|47/49|95.92%|0|
|libero_goal|05|push the plate to the front of the stove|36/37|97.30%|0|
|libero_goal|06|put the cream cheese in the bowl|44/45|97.78%|0|
|libero_goal|07|turn on the stove|48/48|100.00%|0|
|libero_goal|08|put the bowl on the plate|48/50|96.00%|0|
|libero_goal|09|put the wine bottle on the rack|30/41|73.17%|0|
|libero_10|00|put both the alphabet soup and the tomato sauce in the basket|26/49|53.06%|0|
|libero_10|01|put both the cream cheese box and the butter in the basket|12/50|24.00%|0|
|libero_10|02|turn on the stove and put the moka pot on it|45/50|90.00%|0|
|libero_10|03|put the black bowl in the bottom drawer of the cabinet and close it|27/40|67.50%|5|
|libero_10|04|put the white mug on the left plate and put the yellow and white mug on the right plate|36/46|78.26%|0|
|libero_10|05|pick up the book and place it in the back compartment of the caddy|30/36|83.33%|0|
|libero_10|06|put the white mug on the plate and put the chocolate pudding to the right of the plate|27/45|60.00%|0|
|libero_10|07|put both the alphabet soup and the cream cheese box in the basket|26/50|52.00%|0|
|libero_10|08|put both moka pots on the stove|34/43|79.07%|0|
|libero_10|09|put the yellow and white mug in the microwave and close it|32/46|69.57%|2|

源数据：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/summary.json`、`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/plan.json`、`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/independent_final_audit.json`；逐任务统计：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/per_task_summary.json`。本次仅整理已有结果，没有新回放。
