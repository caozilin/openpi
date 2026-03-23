# Normalization statistics

Following common practice, our models normalize the proprioceptive state inputs and action targets during policy training and inference. The statistics used for normalization are computed over the training data and stored alongside the model checkpoint.

## Reloading normalization statistics

When you fine-tune one of our models on a new dataset, you need to decide whether to (A) reuse existing normalization statistics or (B) compute new statistics over your new training data. Which option is better for you depends on the similarity of your robot and task to the robot and task distribution in the pre-training dataset. Below, we list all the available pre-training normalization statistics for each model.

**If your target robot matches one of these pre-training statistics, consider reloading the same normalization statistics.** By reloading the normalization statistics, the actions in your dataset will be more "familiar" to the model, which can lead to better performance. You can reload the normalization statistics by adding an `AssetsConfig` to your training config that points to the corresponding checkpoint directory and normalization statistics ID, like below for the `Trossen` (aka ALOHA) robot statistics of the `pi0_base` checkpoint:

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

For an example of a full training config that reloads normalization statistics, see the `pi0_aloha_pen_uncap` config in the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py).

**Note:** To successfully reload normalization statistics, it's important that your robot + dataset are following the action space definitions used in pre-training. We provide a detailed description of our action space definitions below.

**Note #2:** Whether reloading normalization statistics is beneficial depends on the similarity of your robot and task to the robot and task distribution in the pre-training dataset. We recommend to always try both, reloading and training with a fresh set of statistics computed on your new dataset (see [main README](../README.md) for instructions on how to compute new statistics), and pick the one that works better for your task.


## Provided Pre-training Normalization Statistics

Below is a list of all the pre-training normalization statistics we provide. We provide them for both, the `pi0_base` and `pi0_fast_base` models. For `pi0_base`, set the `assets_dir` to `gs://openpi-assets/checkpoints/pi0_base/assets` and for `pi0_fast_base`, set the `assets_dir` to `gs://openpi-assets/checkpoints/pi0_fast_base/assets`.
| Robot | Description | Asset ID |
|-------|-------------|----------|
| ALOHA | 6-DoF dual arm robot with parallel grippers | trossen |
| Mobile ALOHA | Mobile version of ALOHA mounted on a Slate base | trossen_mobile |
| Franka Emika (DROID) | 7-DoF arm with parallel gripper based on the DROID setup | droid |
| Franka Emika (non-DROID) | Franka FR3 arm with Robotiq 2F-85 gripper | franka |
| UR5e | 6-DoF UR5e arm with Robotiq 2F-85 gripper | ur5e |
| UR5e bi-manual | Bi-manual UR5e setup with Robotiq 2F-85 grippers | ur5e_dual |
| ARX | Bi-manual ARX-5 robot arm setup with parallel gripper | arx |
| ARX mobile | Mobile version of bi-manual ARX-5 robot arm setup mounted on a Slate base | arx_mobile |
| Fibocom mobile | Fibocom mobile robot with 2x ARX-5 arms | fibocom_mobile |


## Pi0 Model Action Space Definitions

Out of the box, both the `pi0_base` and `pi0_fast_base` use the following action space definitions (left and right are defined looking from behind the robot towards the workspace):
```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left arm gripper position",
    "dim_7:dim_12": "right arm joint angles (for bi-manual only)",
    "dim_13": "right arm gripper position (for bi-manual only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (for mobile robots only)",
```

The proprioceptive state uses the same definitions as the action space, except for the base x-y position (the last two dimensions) for mobile robots, which we don't include in the proprioceptive state.

For 7-DoF robots (e.g. Franka), we use the first 7 dimensions of the action space for the joint actions, and the 8th dimension for the gripper action.

General info for Pi robots:
- Joint angles are expressed in radians, with position zero corresponding to the zero position reported by each robot's interface library, except for ALOHA, where the standard ALOHA code uses a slightly different convention (see the [ALOHA example code](../examples/aloha_real/README.md) for details).
- Gripper positions are in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
- Control frequencies are either 20 Hz for UR5e and Franka, and 50 Hz for ARX and Trossen (ALOHA) arms.

For DROID, we use the original DROID action configuration, with joint velocity actions in the first 7 dimensions and gripper actions in the 8th dimension + a control frequency of 15 Hz.

# 归一化统计数据

按照惯例，我们的模型在策略训练和推理期间会对本体感受状态（proprioceptive state）输入和动作目标进行归一化处理。用于归一化的统计数据是基于训练数据计算得出的，并与模型检查点（checkpoint）一起存储。

## 重新加载归一化统计数据

当您在新的数据集上微调我们的任一模型时，您需要决定是 (A) 重用现有的归一化统计数据，还是 (B) 基于新的训练数据计算新的统计数据。哪种选择更适合您，取决于您的机器人和任务与预训练数据集中的机器人和任务分布的相似程度。下面，我们列出了每个模型所有可用的预训练归一化统计数据。

**如果您的目标机器人与这些预训练统计数据中的某一个相匹配，请考虑重新加载相同的归一化统计数据。** 通过重新加载归一化统计数据，您数据集中的动作对模型来说会更加“熟悉”，从而可能带来更好的性能。您可以通过在训练配置中添加一个 `AssetsConfig` 来重新加载归一化统计数据，使其指向相应的检查点目录和归一化统计数据 ID，如下所示，这是针对 `pi0_base` 检查点的 `Trossen`（即 ALOHA）机器人统计数据：

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)

```

关于重新加载归一化统计数据的完整训练配置示例，请参见 [训练配置文件](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py) 中的 `pi0_aloha_pen_uncap` 配置。

**注意：** 为了成功重新加载归一化统计数据，您的机器人和数据集必须遵循预训练中使用的动作空间定义，这一点非常重要。我们在下面提供了对动作空间定义的详细描述。

**注意 #2：** 重新加载归一化统计数据是否有益，取决于您的机器人和任务与预训练数据集中的机器人和任务分布的相似程度。我们建议您始终尝试两种方法：重新加载，以及使用在您的新数据集上计算出的一套全新统计数据进行训练（有关如何计算新统计数据的说明，请参见 [主 README 文件](https://www.google.com/search?q=../README.md)），然后选择对您的任务效果更好的那种方法。

## 提供的预训练归一化统计数据

以下是我们提供的所有预训练归一化统计数据的列表。我们为 `pi0_base` 和 `pi0_fast_base` 模型均提供了这些数据。对于 `pi0_base`，请将 `assets_dir` 设置为 `gs://openpi-assets/checkpoints/pi0_base/assets`；对于 `pi0_fast_base`，请将 `assets_dir` 设置为 `gs://openpi-assets/checkpoints/pi0_fast_base/assets`。

| 机器人 | 描述 | 资产 ID (Asset ID) |
| --- | --- | --- |
| ALOHA | 配备平行夹爪的 6 自由度双臂机器人 | trossen |
| Mobile ALOHA | 安装在 Slate 底座上的移动版 ALOHA | trossen_mobile |
| Franka Emika (DROID) | 基于 DROID 设置、配备平行夹爪的 7 自由度机械臂 | droid |
| Franka Emika (非 DROID) | 配备 Robotiq 2F-85 夹爪的 Franka FR3 机械臂 | franka |
| UR5e | 配备 Robotiq 2F-85 夹爪的 6 自由度 UR5e 机械臂 | ur5e |
| UR5e 双臂 | 配备 Robotiq 2F-85 夹爪的双臂 UR5e 设置 | ur5e_dual |
| ARX | 配备平行夹爪的双臂 ARX-5 机械臂设置 | arx |
| ARX mobile | 安装在 Slate 底座上的移动版双臂 ARX-5 机械臂设置 | arx_mobile |
| Fibocom mobile | 配备 2 个 ARX-5 机械臂的 Fibocom 移动机器人 | fibocom_mobile |

## Pi0 模型动作空间定义

开箱即用地，`pi0_base` 和 `pi0_fast_base` 都使用以下动作空间定义（左和右是站在机器人后方面向工作区来定义的）：

```
    "dim_0:dim_5": "左臂关节角度",
    "dim_6": "左臂夹爪位置",
    "dim_7:dim_12": "右臂关节角度（仅限双臂）",
    "dim_13": "右臂夹爪位置（仅限双臂）",

    # 针对移动机器人：
    "dim_14:dim_15": "底座 x-y 轴速度（仅限移动机器人）",

```

本体感受状态（proprioceptive state）使用与动作空间相同的定义，但移动机器人的底座 x-y 位置（最后两个维度）除外，我们不将其包含在本体感受状态中。

对于 7 自由度机器人（例如 Franka），我们使用动作空间的前 7 个维度表示关节动作，第 8 个维度表示夹爪动作。

Pi 机器人的通用信息：

* 关节角度以弧度（radians）表示，位置零对应于各个机器人接口库报告的零位置，但 ALOHA 除外，标准 ALOHA 代码使用了略微不同的约定（有关详细信息，请参见 [ALOHA 示例代码](https://www.google.com/search?q=../examples/aloha_real/README.md)）。
* 夹爪位置的范围是 [0.0, 1.0]，其中 0.0 代表完全张开，1.0 代表完全闭合。
* 控制频率：UR5e 和 Franka 为 20 Hz，ARX 和 Trossen (ALOHA) 机械臂为 50 Hz。

对于 DROID，我们使用原始的 DROID 动作配置，前 7 个维度为关节速度动作，第 8 个维度为夹爪动作，且控制频率为 15 Hz。

---

Would you like me to translate any other documentation files or help you understand how to implement the `AssetsConfig` in your specific training setup?