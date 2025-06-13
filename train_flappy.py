# === 文件：训练启动主程序 ===
# 位置：E:\ai_work\Flappy Bird\train_flappy.py
# 用途：加载配置，创建 Trainer 实例并启动训练

import os
import sys

# 将项目根目录添加到 Python 路径，确保可以找到 config 和 algorithms 模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# 显式导入 flappy_bird_gymnasium 包，以确保其环境被注册
import flappy_bird_gymnasium # <--- 添加这一行！

from config.training_params import TrainingConfig # 导入配置类
from algorithms.ppo import PPOTrainer # 导入 PPO 训练器

def main():
    # 1. 加载训练配置
    config = TrainingConfig()

    # 2. 根据配置选择并初始化训练器
    # 这里我们只支持 PPO，如果需要支持 DQN，可以 uncomment 对应的部分
    if config.algorithm == "ppo":
        trainer = PPOTrainer(config) # 将配置对象传递给 PPO 训练器
    # elif config.algorithm == "dqn":
    #     from algorithms.dqn import DQNTrainer
    #     trainer = DQNTrainer(config) # 如果实现了DQN，这里可以实例化
    else:
        raise ValueError(f"不支持的算法: {config.algorithm}")

    # 3. 启动训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被用户手动中断。")
        # 此时 trainer 内部已经处理了日志和绘图的保存
    except Exception as e:
        print(f"\n训练过程中发生意外错误: {e}")
        import traceback
        traceback.print_exc()

    # 4. 训练完成后，由 trainer 内部处理绘制图表和测试
    # 不需要在这里再次调用 plot 或 test

if __name__ == "__main__":
    main()