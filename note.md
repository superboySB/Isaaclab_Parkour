# 复现笔记

![](assets/intro.png)

## 配置
在zuanfeng项目的docker内部安装一波环境吧
```sh
cd /workspace/isaaclab

git clone https://github.com/superboySB/Isaaclab_Parkour

cd Isaaclab_Parkour && pip3 install -e .

# 首次使用（或部署到离线服务器）时把 GO2 机器人完整资产拉到本地（只需执行一次），需要宿主机网络环境执行
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
