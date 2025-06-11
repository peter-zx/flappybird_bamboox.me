# === 修改文件：训练脚本 ===
# 原因：优化日志记录和绘图，支持算法切换
from startup.init_trainer import initialize_trainer
from config.training_params import TrainingConfig
from plot_utils import plot_training_progress  # 新增：导入绘图工具
import os
import json
import torch

# === 修改部分：设置路径 ===
# 原因：统一路径管理，保存日志和图表
PROJECT_ROOT = r"E:\ai_work\Flappy Bird"
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
# === 修改结束 ===

def main():
    # 初始化训练器
    config = TrainingConfig()
    config.algorithm = "ppo"  # 可改为 "dqn" 切换算法
    trainer = initialize_trainer(config)

    # 加载保存点
    save_path = os.path.join(PROJECT_ROOT, "flappy_bird_models_20250610_184044", "model_step_1007616.pth")
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        trainer.network.load_state_dict(checkpoint['network_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.episode_rewards = checkpoint['episode_rewards']
        trainer.episode_lengths = checkpoint['episode_lengths']
        print(f"从 {save_path} 加载模型，步数: {checkpoint['timesteps']}")
    
    # === 修改部分：初始化日志 ===
    # 原因：保存训练数据，便于分析和绘图
    train_log_path = os.path.join(LOG_DIR, "train_log.json")
    train_results = []
    # === 修改结束 ===
    
    # 开始训练
    try:
        save_dir = trainer.train(total_timesteps=config.total_timesteps, start_timesteps=config.start_timesteps)
        print(f"\n训练完成! 模型保存在: {save_dir}")
        
        # === 修改部分：保存训练日志 ===
        # 原因：记录奖励和步数数据
        train_results = [
            {"episode": i + 1, "reward": r, "steps": l}
            for i, (r, l) in enumerate(zip(trainer.episode_rewards, trainer.episode_lengths))
        ]
        with open(train_log_path, "w") as f:
            json.dump(train_results, f, indent=4)
        # === 修改结束 ===
        
        # === 修改部分：绘制训练进度 ===
        # 原因：优化绘图，与 training_progress.png 一致
        trainer.plot_training_progress()  # 保留原始绘图
        plot_training_progress(
            trainer.episode_rewards,
            trainer.episode_lengths,
            save_path=os.path.join(PLOT_DIR, "train_progress.png")
        )
        print(f"训练进度图已保存至: {PLOT_DIR}\\train_progress.png")
        # === 修改结束 ===
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        trainer.plot_training_progress()
        # 保存日志
        train_results = [
            {"episode": i + 1, "reward": r, "steps": l}
            for i, (r, l) in enumerate(zip(trainer.episode_rewards, trainer.episode_lengths))
        ]
        with open(train_log_path, "w") as f:
            json.dump(train_results, f, indent=4)
        plot_training_progress(
            trainer.episode_rewards,
            trainer.episode_lengths,
            save_path=os.path.join(PLOT_DIR, "train_progress.png")
        )
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()