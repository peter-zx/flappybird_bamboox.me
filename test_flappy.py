# === 文件：测试脚本 ===
# 位置：E:\ai_work\Flappy Bird\test_flappy.py
# 用途：加载训练好的模型，并在环境中进行评估测试

import os
import sys

# 将项目根目录添加到 Python 路径，确保可以找到 config 和 algorithms 模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# 显式导入 flappy_bird_gymnasium 包，以确保其环境被注册
import flappy_bird_gymnasium # <--- 添加这一行！

from config.training_params import TrainingConfig
from algorithms.ppo import PPOTrainer # 导入 PPO 训练器

def main():
    # 1. 加载训练配置
    config = TrainingConfig()

    # 2. 设置测试时的渲染模式
    # 如果要看游戏画面，这里设置为 "human"
    config.test_render_mode = "human" 
    # 测试时并行环境数量，通常测试单个环境即可
    config.n_envs = 1 # 测试时通常只需要一个环境

    # 3. 指定要加载的模型路径
    # 如果 config.load_model_path 为 None，则自动查找最新训练的模型
    if config.load_model_path is None:
        # 构建模型保存的绝对根目录路径 (例如：E:\ai_work\Flappy Bird\trained_models)
        base_model_save_path = os.path.join(PROJECT_ROOT, config.model_save_base_dir)

        if not os.path.exists(base_model_save_path):
            print(f"错误：模型保存根目录 '{base_model_save_path}' 不存在。请先运行训练以生成模型文件，或手动在 config/training_params.py 中指定 load_model_path。")
            return

        # 列出模型保存根目录下的所有子目录 (例如：flappy_bird_models_YYYYMMDD_HHMMSS)
        all_model_dated_dirs = [d for d in os.listdir(base_model_save_path) 
                                if os.path.isdir(os.path.join(base_model_save_path, d)) and d.startswith("flappy_bird_models_")]

        if not all_model_dated_dirs:
            print(f"错误：在 '{base_model_save_path}' 中未找到任何训练好的模型目录。请先运行训练或手动指定模型路径。")
            return

        # 找到最新的带时间戳的目录
        latest_model_dir_name = sorted(all_model_dated_dirs)[-1] 
        latest_model_dir_full_path = os.path.join(base_model_save_path, latest_model_dir_name)

        # 在最新模型目录下查找最新的 .pth 文件
        latest_checkpoint_file = None
        latest_step = -1
        for f_name in os.listdir(latest_model_dir_full_path):
            if f_name.startswith("model_step_") and f_name.endswith(".pth"):
                try:
                    step = int(f_name.split('_')[-1].split('.')[0])
                    if step > latest_step:
                        latest_step = step
                        latest_checkpoint_file = os.path.join(latest_model_dir_full_path, f_name)
                except ValueError:
                    continue # 忽略不符合命名规范的文件

        if latest_checkpoint_file:
            config.load_model_path = latest_checkpoint_file
            print(f"自动检测到并加载最新模型：{config.load_model_path}")
        else:
            print(f"错误：在最新模型目录 '{latest_model_dir_full_path}' 中未找到任何模型文件。请先运行训练或手动指定模型路径。")
            return

    # 4. 初始化训练器 (它会根据 load_model_path 加载模型)
    if config.algorithm == "ppo":
        trainer = PPOTrainer(config)
    # elif config.algorithm == "dqn":
    #     from algorithms.dqn import DQNTrainer
    #     trainer = DQNTrainer(config)
    else:
        raise ValueError(f"不支持的算法: {config.algorithm}")

    # 5. 调用 Trainer 内部的测试方法
    trainer.test_agent(config.load_model_path)

if __name__ == "__main__":
    main()