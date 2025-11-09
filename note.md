# 复现笔记

![](assets/intro.png)

## 配置
在zuanfeng项目的docker内部安装一波环境吧
```sh
cd /workspace/isaaclab

git clone https://github.com/superboySB/Isaaclab_Parkour

cd Isaaclab_Parkour && pip3 install -e .

# 首次使用（或部署到离线服务器）时把 GO2 机器人完整资产（含天空 HDR、瓷砖材质、UI Arrow 等）拉到本地，需在可联网环境执行一次
python scripts/tools/download_go2_assets.py

cd parkour_tasks && pip3 install -e .
```

默认跳过已存在文件，如需重新下载可加 `--force`。下载完成后，`play/eval/demo/train` 都会自动引用本地 `assets/nucleus/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/Go2/` 目录，无需联网。
```

## 直接看预训练结果
`assets/` 目录里已经下好了对应 checkpoint：
- `assets/pretrained_teacher/model_*.pt`
- `assets/pretrained_student/model_*.pt`

所有非训练脚本（play / evaluation / demo，以及学生模型的蒸馏验证）都支持两种方式加载权重：
- 直接加 `--use_pretrained_checkpoint`，强制使用 `assets/` 下最新的 `model_*.pt`。
- 如果已经有重新训练得到的日志，可以用 `--log_root /path/to/logs/rsl_rl/unitree_go2_parkour` 指定日志目录而不加 `--use_pretrained_checkpoint`。

Teacher 预训练结果（直接用 assets）：
```sh
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 1 --use_pretrained_checkpoint

python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0 --use_pretrained_checkpoint

python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --use_pretrained_checkpoint
```

Student（蒸馏）预训练结果（直接用 assets）：
```sh
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 1 --use_pretrained_checkpoint

python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 --use_pretrained_checkpoint

python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --use_pretrained_checkpoint
```

其中，play就是可视化，evaluation就是多次测试输出metrics，而demo基于`play`的加载逻辑，却额外创建相机、手柄控制、第三人称视角等交互能力，适合手动控制。要切换到自己训练得到的日志，只需要把 `--use_pretrained_checkpoint` 替换成
`--log_root /workspace/isaaclab/Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour`（或其他日志路径），脚本会从该目录里自动挑最新 checkpoint。

**深度图调试**：学生任务在 PLAY/EVAL（以及 demo）配置里自动开启深度调试窗口，方便本地预览；训练任务（`--task ...Student-Unitree-Go2-v0`）默认关闭该窗口以便在无显示的服务器上运行。如果需要放大调试视图，可设置 `DEPTH_DEBUG_VIEWER_SCALE=6 python scripts/rsl_rl/play.py ...`。

上面这些命令会保持现有场景/训练参数不变，只是载入指定 checkpoint，运行后也会在相应日志目录下自动导出 JIT/ONNX（play.py 内置的导出逻辑）。

## 重新训练两阶段
1. **Teacher：先训老师拿到参考权重**  
   ```sh
   python scripts/rsl_rl/train.py \
     --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
     --seed 1 --headless \
     --run_name teacher_seed1
   ```  
   - 日志会保存在 `logs/rsl_rl/unitree_go2_parkour/<timestamp>_teacher_seed1/`。  
   - 训练过程中每 `save_interval(=100)` 轮会生成 `model_<iter>.pt`，结束时还会写入 `model_50000.pt`（最后一次迭代）。脚本不会自动挑 “最优” checkpoint，需要你根据 `evaluation.py` 的指标手动挑一个，例如 `model_48000.pt` 或 `model_50000.pt`。  
   - 对任意 checkpoint 运行 `python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0 --checkpoint logs/rsl_rl/unitree_go2_parkour/<run>/model_50000.pt` 来验证。

2. **Student：蒸馏阶段需要加载老师 checkpoint**  
   ```sh
   python scripts/rsl_rl/train.py \
     --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 \
     --seed 1 --headless \
     --run_name student_seed1 \
     --checkpoint /workspace/isaaclab/Isaaclab_Parkour/assets/pretrained_teacher/model_49999.pt
   ```  
   - 现在可以直接把老师 `model_*.pt` 的绝对路径传给 `--checkpoint`，脚本会自动判定并加载，不再强制要求 `--load_run`。如果仍想用旧方式指定目录，也可以保留 `--load_run` + `--checkpoint model_50000.pt` 的组合。  
   - 学生配置的算法是 `DistillationWithExtractor`，`train.py` 会在启动新日志前先加载这个老师 checkpoint，然后再开始学生训练。后续学生的 checkpoint 同样写到 `logs/rsl_rl/unitree_go2_parkour/<timestamp>_student_seed1/model_*.pt`。

3. **评估或导出**  
   - 训完老师或学生后，使用 `play.py / evaluation.py / demo.py` 并通过 `--log_root` + `--checkpoint` 指向这些新日志，就能直接在本地查看表现，步骤与前述“预训练结果”一致。

> 说明：训练脚本默认关闭 git diff 记录和任何联网操作，日志只写入本地的 `logs/rsl_rl/...`。如需恢复 git 状态快照，可手动设置 `ISAACLAB_ENABLE_GIT_STATE=1` 再运行。

### 原作者的todo
一个sim2sim的pipeline，对应mujoco：
https://github.com/CAI23sbP/go2_parkour_deploy

## 代码理解

> 目标：快速掌握老师（强化学习）与学生（蒸馏）两个阶段的训练入口、观测/奖励维度、动作接口与网络架构。源码分布主要在 `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2` 与 `scripts/rsl_rl/modules`。

### 阶段一：Teacher（强化学习）

#### 训练入口与场景
- 命令：`python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless`.
- 场景：`ParkourTeacherSceneCfg`（`parkour_teacher_cfg.py`）创建 6144 并行环境、0.005 s 物理步长、20 s episode，启用 RayCaster 高度扫描器与 ContactSensor。
- 随机事件：`parkour_mdp_cfg.py` 中的 `EventCfg` 会在 reset/startup/interval 注入关节随机化、摩擦/质量/质心扰动、相机姿态扰动、每 8 s 推力等，防止过拟合。

#### 指令与动作
- 指令生成：`ParkourCommandCfg` 每 6 s 采样前向速度 0.3~0.8 m/s、目标航向 -1.6~1.6 rad，yaw 控制通过比例环裁剪到 ±1 rad（`uniform_parkour_command.py`）。
- 动作通道：`DelayedJointPositionActionCfg` 控制全部 12 关节位置偏置；老师阶段设置 `use_delay=False`、历史长度 1，相当于直接输出关节目标（`parkour_teacher_cfg.py:58`）。

#### 观测维度速查

| 组成 | 维度 | 说明 / 代码 |
| --- | --- | --- |
| 本体 `prop` | 53 | `root_ang_vel_b`(3)、IMU roll/pitch(2)、保留位(1)、`delta_yaw`(1) + `delta_next_yaw`(1)、指令占位(2)+ `cmd_vx`(1)、地形类型 one-hot(2)、12 关节位置偏差 + 12 速度(缩放 0.05)、上一帧动作 12、脚接触 4（`observations.py:72`） |
| 高度射线 | 132 | Base 上方 RayCaster 探测的地形高度差（`observations.py:90`） |
| 显式特权 | 9 | 根线速度×2 + 6 个零填充，代表可测 IMU/速度信息（`observations.py:117`） |
| 隐式特权 | 29 | 基座质量/质心(4)、摩擦系数(1)、各关节刚度/阻尼缩放各 12（`observations.py:124`） |
| 历史缓冲 | 530 | 最近 10 帧 prop（10×53），由 `StateHistoryEncoder` 提取 latent（`observations.py:95`） |

#### 奖励速查

| 名称 | 权重 | 含义（函数） |
| --- | --- | --- |
| `reward_collision` | -10 | 机体/大腿/小腿接触惩罚（`rewards.py:215`） |
| `reward_feet_edge` | -1 | 脚落在台阶边缘并接触（`rewards.py:19`）|
| `reward_torques` / `reward_delta_torques` | -1e-5 / -1e-7 | 扭矩及其变化 L2 惩罚（`rewards.py:66`,`196`）|
| `reward_dof_error` / `reward_hip_pos` | -0.04 / -0.5 | 关节偏差、髋姿势偏差（`rewards.py:73`,`80`）|
| `reward_ang_vel_xy` | -0.05 | 基座横向角速度平方（`rewards.py:96`）|
| `reward_action_rate` | -0.1 | 动作跃变惩罚（`rewards.py:107`）|
| `reward_dof_acc` | -2.5e-7 | 关节角加速度平方（`rewards.py:121`）|
| `reward_lin_vel_z` / `reward_orientation` | -1 / -1 | 垂直速度、姿态偏差（`rewards.py:135`,`147`）|
| `reward_feet_stumble` | -1 | 足端横向力突变，视为绊倒（`rewards.py:159`）|
| `reward_tracking_goal_vel` | +1.5 | 速度投影到目标方向的比值（`rewards.py:169`）|
| `reward_tracking_yaw` | +0.5 | 指数形式的 yaw 误差（`rewards.py:184`）|

终止条件包括：到达所有目标、episode 超时、roll/pitch 超 ±1.5 rad 或高度低于 -0.25 m（`terminations.py:25`）。

#### 神经网络与算法
- **ActorCriticRMA**：Actor 输入为 `[prop, scan_latent, priv_explicit, latent]`，三层 `[512,256,128]` MLP 输出关节目标。Scan encoder `[128,64,32]`，History encoder `StateHistoryEncoder` 以 1D Conv 处理 10×53 缓冲输出与 `priv_encoder_dims=[64,20]` 匹配的 latent（`actor_critic_with_encoder.py`）。Critic 同结构处理特权观测。
- **DefaultEstimator**：`num_prop=53 -> hidden [128,64] -> 9`，估计物理特征并在训练时替换进入观测（`feature_extractors/estimator.py`，`ppo_with_extractor.py:82`）。
- **PPOWithExtractor**：24 步回合、5 epoch ×4 mini-batch、γ=0.99、λ=0.95、clip=0.2、entropy=0.01、自适应 KL 目标 0.01。额外损失：历史 latent 与特权 latent L2 (`priv_reg_coef_schedual=[0,0.1,2000,3000]`) 以及估计器 MSE（`ppo_with_extractor.py:183-210`）。

### 阶段二：Student（视觉蒸馏）

#### 训练入口与场景
- 命令：`python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --checkpoint <teacher_model>`.
- 场景：`ParkourStudentSceneCfg` 把地形网格扩展到 10×20，`horizontal_scale=0.1`，多数子地形 `use_simplified=True`。训练环境 192 个（Eval 256、Play 16），并挂载 `camera_cfg` 深度相机，Play/Eval 再附加 USD 可视化（`parkour_student_cfg.py`）。

#### 观测与动作
- 本体观测：`StudentObservationsCfg.policy` 仍使用 `ExtremeParkourObservations`，维度与老师完全一致（见上表）。
- 额外观测：

| 观测组 | 维度/配置 | 用途 |
| --- | --- | --- |
| `depth_camera` | 58×87 深度图，缓冲 2 帧；可选 `debug_vis`（`observations.py:145`） | 提供视觉输入给蒸馏用的深度编码器 |
| `delta_yaw_ok` | bool mask | 判断 `|target_yaw - yaw| < 0.6` 时允许视觉预测覆盖真值（`observations.py:275`） |

- 动作延迟：学生启用 `use_delay=True`、`history_length=8`，`action_delay_steps=[1,1]`（`parkour_mdp_cfg.py:338`）。`DelayedJointPositionAction` 会在 8 帧队列中回溯 1 步后才下发，模拟执行链路延时。

#### 奖励
- 训练配置仅保留 `reward_collision` 且权重设为 0（`parkour_mdp_cfg.py:106`），真正的学习信号来自蒸馏损失。
- Eval/Play 则切换到 `TeacherRewardsCfg` 以便评估指标与老师一致（`parkour_student_cfg.py:73`）。

#### 蒸馏网络
- **DepthOnlyFCBackbone58x87** → Conv5×5 → MaxPool → Conv3×3 → Flatten → Linear(128) → Linear 输出 32 维 latent（`feature_extractors/depth_backbone.py:4`）。
- **RecurrentDepthBackbone**：将 latent 与 53 维 prop 拼接经 MLP（128→32）并送入 GRU(hidden=512)，输出 34 维，其中 32 维为 `scandots_latent`，最后两维（tanh×1.5）预测 `delta_yaw` 与 `delta_next_yaw`（`depth_backbone.py:33`）。
- **Depth Actor**：老师 Actor 的深拷贝，仅更新与视觉编码器连接的参数（`distillation_with_extractor.py:50`）。

#### DistillationWithExtractor 流程（`on_policy_runner_with_extractor.py`）
1. **准备**：加载老师 policy (`self.alg.policy`) 与学生 depth actor (`self.alg.depth_actor`)，前者只做推理。深度编码器设置为训练模式，并统计 `delta_yaw_ok` 覆盖。
2. **视觉前向**：每当 `common_step_counter % 5 == 0`，复制当前 prop 并将 yaw 槽位清零，再送入深度编码器得到 `[latent, yaw_pred]`。这样视觉网络必须依靠深度数据推断 yaw。
3. **Teacher vs Student**：
   - 老师：`actions_teacher = policy.act_inference(obs, hist_encoding=True)`。
   - 学生：若 `delta_yaw_ok` 为真，则把 `obs[:,6:8]` 替换为视觉预测，然后用 depth actor（输入 obs + depth latent）输出 `actions_student`。
4. **缓冲损失**：累计 `actions_buffer.append(actions_teacher - actions_student)`、`yaws_buffer.append(obs[:,6:8] - yaw_pred)`；同时记录 `delta_yaw_ok` 覆盖率。
5. **环境推进**：默认 `num_pretrain_iter=0`，即直接用学生动作驱动环境；可按需修改以便先用老师动作热身。
6. **参数更新**：每次迭代结束调用 `update_depth_actor`，损失 `L = mean(||Δa||_2) + mean(||Δyaw||_2)`，学习率 1e-3，梯度裁剪 `max_grad_norm=1`。更新完成后 `detach_hidden_states()` 防止 GRU 梯度积累。
7. **日志**：记录 `depth_actor_loss`, `yaw_loss`, `delta_yaw_ok_percentage` 以及采样/学习耗时，方便监控蒸馏收敛。

#### 关系总结
- 老师与学生共享相同的 RMA 结构；蒸馏阶段冻结老师 ActorCriticRMA，仅优化学生 depth actor + 深度编码器。
- `DefaultEstimator` 依旧存在于 `DistillationWithExtractor` 配置内，为算法提供 `num_prop/num_priv_explicit/num_scan` 等尺寸信息，并可继续进行特征估计监督。

> 结论：老师阶段通过大规模 PPO + RMA + 特权观测学习出可靠策略；学生阶段加载老师权重、叠加动作延迟与深度相机，依靠 L2 蒸馏损失把动作与偏航判断迁移到视觉驱动的低特权模型，从而在真实部署更具可行性。
