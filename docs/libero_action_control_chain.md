# LIBERO env.step(action) 完整调用链分析

## 调用链总览

```
OffScreenRenderEnv.step(action)          # 环境包装
  → BDDLBaseDomain.step(action)          # LIBERO 任务基类
    → MujocoEnv.step(action)             # robosuite 基类，核心仿真循环
      → 循环 25 次 (control_timestep / model_timestep):
          sim.forward()                  # 前向运动学
          _pre_action(action)            # 控制器更新
            → robot.control(action)      # SingleArm 分解动作
              → controller.set_goal()    # 设置笛卡尔空间目标
              → controller.run_controller()  # OSC 算法 → 关节力矩
              → sim.data.ctrl = torques  # 写入仿真器
          sim.step()                     # mujoco.mj_step() 物理仿真
```

---

## 1. Action 的含义（7维）

```
action[0:3]  → delta_xyz   位置增量，[-1,1] 缩放到 [-0.05m, 0.05m]
action[3:6]  → delta_rot   旋转增量，[-1,1] 缩放到 [-0.5rad, 0.5rad]
action[6]    → gripper     夹爪，-1=打开，1=关闭
```

在 `SingleArm.control()` 中分解为：
- `arm_action = action[:6]` → 给 OSC 控制器
- `gripper_action = action[6:]` → 给夹爪

配置来源：`robosuite/controllers/config/osc_pose.json`

---

## 2. 环境层

### OffScreenRenderEnv

- **文件**: `libero/libero/envs/env_wrapper.py`
- 继承自 `ControlEnv`，用于离屏渲染和评估
- 默认控制器: `OSC_POSE`

### ControlEnv.step

```python
def step(self, action):
    return self.env.step(action)
```

### BDDLBaseDomain.step

- **文件**: `libero/libero/envs/bddl_base_domain.py`
- 继承自 `SingleArmEnv`，所有 LIBERO 任务的基础类

```python
def step(self, action):
    if self.action_dim == 4 and len(action) > 4:
        # 转换 OSC_POSITION action (只有位置控制)
        action = np.array(action)
        action = np.concatenate((action[:3], action[-1:]), axis=-1)

    obs, reward, done, info = super().step(action)
    done = self._check_success()
    return obs, reward, done, info
```

---

## 3. MuJoCo 环境基类

### MujocoEnv.step — 核心仿真循环

- **文件**: `robosuite/environments/base.py`

```python
def step(self, action):
    self.timestep += 1
    policy_step = True

    # 关键循环: 在两个策略步之间执行多次仿真步
    for i in range(int(self.control_timestep / self.model_timestep)):
        self.sim.forward()                    # 前向运动学
        self._pre_action(action, policy_step) # 控制器更新
        self.sim.step()                       # 物理仿真步进
        self._update_observables()            # 更新观测
        policy_step = False

    self.cur_time += self.control_timestep
    reward, done, info = self._post_action(action)
    observations = self._get_observations()
    return observations, reward, done, info
```

**时间参数**:
- `control_freq = 20 Hz` → `control_timestep = 0.05s`
- `model_timestep = 0.002s`（来自 `robosuite/macros.py`）
- 每个 action 执行 `0.05 / 0.002 = 25` 次 MuJoCo 仿真步

### RobotEnv._pre_action

- **文件**: `robosuite/environments/robot_env.py`

```python
def _pre_action(self, action, policy_step=False):
    assert len(action) == self.action_dim

    cutoff = 0
    for idx, robot in enumerate(self.robots):
        robot_action = action[cutoff : cutoff + robot.action_dim]
        robot.control(robot_action, policy_step=policy_step)
        cutoff += robot.action_dim
```

---

## 4. 机器人控制层

### SingleArm.control — action 处理和控制器调用

- **文件**: `robosuite/robots/single_arm.py`

```python
def control(self, action, policy_step=False):
    assert len(action) == self.action_dim

    # 分离夹爪动作和手臂动作
    gripper_action = None
    if self.has_gripper:
        gripper_action = action[self.controller.control_dim:]  # 最后1维
        arm_action = action[: self.controller.control_dim]     # 前6维 (OSC_POSE)
    else:
        arm_action = action

    # 更新控制器目标 (仅在新策略步时)
    if policy_step:
        self.controller.set_goal(arm_action)

    # 运行控制器计算关节力矩
    torques = self.controller.run_controller()

    # 力矩裁剪
    low, high = self.torque_limits
    self.torques = np.clip(torques, low, high)

    # 设置夹爪动作
    if self.has_gripper:
        self.grip_action(gripper=self.gripper, gripper_action=gripper_action)

    # 将力矩写入仿真器
    self.sim.data.ctrl[self._ref_joint_actuator_indexes] = self.torques

    # 更新缓冲区 (仅在策略步时)
    if policy_step:
        self.recent_qpos.push(self._joint_positions)
        self.recent_actions.push(action)
        self.recent_torques.push(self.torques)
```

---

## 5. OSC 控制器（核心算法）

### 概述

- **文件**: `robosuite/controllers/osc.py`
- 基于论文: [Khatib 1987 - Unified Approach to Motion and Force Control](http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf)

### 参数

```python
kp = [150, 150, 150, 150, 150, 150]     # 位置和方向增益
kd = 2 * √kp ≈ [24.5, ...]              # 阻尼增益 (临界阻尼)
uncouple_pos_ori = True                  # 解耦位置和方向控制
control_delta = True                     # 相对控制 (delta 模式)
```

### set_goal — 将 delta action 转为笛卡尔目标

```python
def set_goal(self, action, set_pos=None, set_ori=None):
    self.update()  # 更新机器人状态

    delta = action  # "fixed" impedance 模式
    scaled_delta = self.scale_action(delta)  # 缩放到实际物理单位

    # 设置目标位置和方向
    self.goal_pos = set_goal_position(
        scaled_delta[:3], self.ee_pos,
        position_limit=self.position_limits, set_pos=set_pos
    )
    self.goal_ori = set_goal_orientation(
        scaled_delta[3:6], self.ee_ori_mat,
        orientation_limit=self.orientation_limits, set_ori=set_ori
    )
```

### run_controller — 核心 OSC 算法

```python
def run_controller(self):
    self.update()  # 更新状态

    # 1. 获取期望位置 (可带插值)
    desired_pos = self.interpolator_pos.get_interpolated_goal() \
        if self.interpolator_pos else self.goal_pos

    # 2. 计算方向误差
    desired_ori = self.goal_ori
    ori_error = orientation_error(desired_ori, self.ee_ori_mat)

    # 3. 计算期望力 (PD 控制)
    position_error = desired_pos - self.ee_pos
    vel_pos_error = -self.ee_pos_vel
    desired_force = kp[0:3] * position_error + kd[0:3] * vel_pos_error

    # 4. 计算期望力矩 (PD 控制)
    vel_ori_error = -self.ee_ori_vel
    desired_torque = kp[3:6] * ori_error + kd[3:6] * vel_ori_error

    # 5. 计算操作空间矩阵
    lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
        self.mass_matrix, self.J_full, self.J_pos, self.J_ori
    )

    # 6. 解耦位置和方向控制
    if self.uncoupling:
        decoupled_force = lambda_pos @ desired_force
        decoupled_torque = lambda_ori @ desired_torque
        decoupled_wrench = [decoupled_force, decoupled_torque]
    else:
        desired_wrench = [desired_force, desired_torque]
        decoupled_wrench = lambda_full @ desired_wrench

    # 7. 笛卡尔 → 关节空间
    self.torques = J_full.T @ decoupled_wrench + self.torque_compensation

    # 8. 零空间力矩 (保持初始关节姿态)
    self.torques += nullspace_torques(
        self.mass_matrix, nullspace_matrix,
        self.initial_joint, self.joint_pos, self.joint_vel
    )

    return self.torques
```

### Controller.update — 更新机器人状态

- **文件**: `robosuite/controllers/base_controller.py`

```python
def update(self, force=False):
    if self.new_update or force:
        self.sim.forward()

        # 末端执行器状态
        self.ee_pos = sim.data.site_xpos[eef_site_id]
        self.ee_ori_mat = sim.data.site_xmat[eef_site_id].reshape(3,3)
        self.ee_pos_vel = sim.data.get_site_xvelp(eef_name)
        self.ee_ori_vel = sim.data.get_site_xvelr(eef_name)

        # 关节状态
        self.joint_pos = sim.data.qpos[qpos_index]
        self.joint_vel = sim.data.qvel[qvel_index]

        # 雅可比矩阵
        self.J_pos = sim.data.get_site_jacp(eef_name)[qvel_index]  # 3×7
        self.J_ori = sim.data.get_site_jacr(eef_name)[qvel_index]  # 3×7
        self.J_full = vstack([J_pos, J_ori])                       # 6×7

        # 质量矩阵
        mujoco.mj_fullM(model, mass_matrix, sim.data.qM)
        self.mass_matrix = mass_matrix[qvel_index][:, qvel_index]  # 7×7
```

---

## 6. 辅助数学函数

- **文件**: `robosuite/utils/control_utils.py`

### opspace_matrices — 操作空间矩阵计算

```python
def opspace_matrices(mass_matrix, J_full, J_pos, J_ori):
    M_inv = inv(mass_matrix)

    # lambda 矩阵 (操作空间惯性矩阵的逆)
    lambda_full_inv = J_full @ M_inv @ J_full.T
    lambda_pos_inv = J_pos @ M_inv @ J_pos.T
    lambda_ori_inv = J_ori @ M_inv @ J_ori.T

    # 求逆 (使用伪逆保证数值稳定)
    lambda_full = pinv(lambda_full_inv)
    lambda_pos = pinv(lambda_pos_inv)
    lambda_ori = pinv(lambda_ori_inv)

    # 零空间投影矩阵
    Jbar = M_inv @ J_full.T @ lambda_full  # 动力学一致雅可比逆
    nullspace_matrix = I - Jbar @ J_full   # N = I - J̄*J

    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix
```

### nullspace_torques — 零空间力矩计算

```python
def nullspace_torques(mass_matrix, nullspace_matrix,
                      initial_joint, joint_pos, joint_vel, joint_kp=10):
    joint_kv = sqrt(joint_kp) * 2  # 临界阻尼

    # PD 控制回到初始姿态
    pose_torques = mass_matrix @ (
        joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel
    )

    # 投影到零空间
    nullspace_torques = nullspace_matrix.T @ pose_torques
    return nullspace_torques
```

### orientation_error — 方向误差计算

```python
def orientation_error(desired, current):
    # 使用 Khatib 公式，轴角表示的方向误差
    error = 0.5 * (cross(rc1, rd1) + cross(rc2, rd2) + cross(rc3, rd3))
    return error  # 3D 向量
```

---

## 7. MuJoCo 仿真层

- **文件**: `robosuite/utils/binding_utils.py`

### MjSim.step

```python
def step(self, with_udd=True):
    mujoco.mj_step(self.model._model, self.data._data)
```

`mj_step` 内部执行两个半步:
- **mj_step1**: 计算加速度和力（正向运动学、雅可比、惯性、约束力、加速度 `M * q̈ = τ + τ_passive - J^T * f_constraint`）
- **mj_step2**: 时间积分更新位置和速度（`q̇ ← q̇ + q̈ * dt`，`q ← q + q̇ * dt`）

### MjSim.forward

```python
def forward(self):
    mujoco.mj_forward(self.model._model, self.data._data)
```

计算正向运动学：位置、速度、加速度、雅可比矩阵、惯性矩阵等。

---

## 8. 核心数学公式

### PD 控制

```
F_desired = Kp_pos * (x_goal - x_current) + Kd_pos * (-ẋ)
τ_desired = Kp_ori * orientation_error(R_goal, R_current) + Kd_ori * (-ω)
```

### 操作空间惯性矩阵

```
Λ = (J * M⁻¹ * Jᵀ)⁻¹
```

### 解耦控制

```
F_decoupled = Λ_pos * F_desired
τ_decoupled = Λ_ori * τ_desired
```

### 笛卡尔 → 关节空间

```
τ_main = Jᵀ @ [F_decoupled, τ_decoupled] + τ_gravity
```

### 零空间力矩

```
J̄ = M⁻¹ * Jᵀ * Λ           (动力学一致雅可比逆)
N = I - J̄ * J               (零空间投影矩阵)
τ_null = Nᵀ * M * (Kp*(q_init - q) - Kd*q̇)
```

### 最终力矩

```
τ_final = τ_main + τ_null
```

---

## 9. 关键数学量维度

| 符号 | 含义 | 维度 | 说明 |
|------|------|------|------|
| `action` | 输入动作 | 7×1 | 策略网络输出 |
| `arm_action` | 手臂动作 | 6×1 | action[:6] |
| `gripper_action` | 夹爪动作 | 1×1 | action[6] |
| `goal_pos` | 目标位置 | 3×1 | 笛卡尔空间 |
| `goal_ori` | 目标方向 | 3×3 | 旋转矩阵 |
| `position_error` | 位置误差 | 3×1 | Δx |
| `ori_error` | 方向误差 | 3×1 | 轴角形式 |
| `F_desired` | 期望力 | 3×1 | PD 控制输出 |
| `τ_desired` | 期望力矩 | 3×1 | PD 控制输出 |
| `J_pos` | 位置雅可比 | 3×7 | ∂x/∂q |
| `J_ori` | 方向雅可比 | 3×7 | ∂θ/∂q |
| `J_full` | 完整雅可比 | 6×7 | [J_pos; J_ori] |
| `M` | 质量矩阵 | 7×7 | 关节空间惯性 |
| `Λ_pos` | 位置操作空间惯性 | 3×3 | (J M⁻¹ Jᵀ)⁻¹ |
| `Λ_ori` | 方向操作空间惯性 | 3×3 | (J M⁻¹ Jᵀ)⁻¹ |
| `N` | 零空间投影 | 7×7 | I - J̄J |
| `torques` | 关节力矩 | 7×1 | 最终控制输出 |

> Panda 机器人有 7 个关节自由度 (DOF)，是冗余机械臂。

---

## 10. 完整数据流总结

```
Input Action (7D, 归一化 [-1, 1])
  ↓
[分解] arm_action (6D) | gripper_action (1D)
  ↓                          ↓
[缩放] scaled_delta       [夹爪控制]
  ↓
[目标设置] goal_pos (3D), goal_ori (3×3 矩阵)
  ↓
[PD 控制] F_desired (3D), τ_desired (3D)
  ↓
[操作空间映射]
  ├─ 计算 Λ_pos, Λ_ori (操作空间惯性)
  ├─ 解耦: F_decoupled, τ_decoupled
  └─ 笛卡尔 → 关节: τ_main = Jᵀ @ wrench
  ↓
[零空间控制] τ_null (保持初始姿态)
  ↓
[合成] τ_final = τ_main + τ_null + τ_gravity
  ↓
[裁剪] clip(τ_final, torque_limits)
  ↓
[应用] sim.data.ctrl[0:7] = τ_final
  ↓
[MuJoCo 仿真]
  ├─ mj_step1: 计算加速度 q̈
  └─ mj_step2: 积分更新 q, q̇
  ↓
[重复 25 次] (每个 action)
  ↓
Output: obs, reward, done, info
```
