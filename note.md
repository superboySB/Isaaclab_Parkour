# 复现笔记

![](assets/intro.png)

## 配置
在zuanfeng项目的docker内部安装一波环境吧
```sh
cd /workspace/isaaclab

git clone https://github.com/superboySB/Isaaclab_Parkour

cd Isaaclab_Parkour && pip3 install -e .

cd parkour_tasks && pip3 install -e .
```

## 直接看预训练结果
已经在assets下好了teacher模型，可以直接测试teacher
```sh
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 16
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0 
```
其实也已经在assets下好了student模型，可以直接测试student
```sh
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 
```

## 重新训练两阶段
```sh
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless
```

## 部署
```sh
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 
```

## Tips
### 相机控制按键
```
press 1 or 2: Going to environment

press 8: camera forward    

press 4: camera leftward   

press 6: camera rightward   

press 5: camera backward

press 0: Use free camera (can use mouse)

press 1: Not use free camera (default)
```

### 原作者的todo
一个sim2sim的pipeline，对应mujoco：
https://github.com/CAI23sbP/go2_parkour_deploy