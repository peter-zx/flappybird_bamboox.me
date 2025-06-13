# === 修改文件：训练脚本 ===
# 原因：优化日志记录和绘图，支持算法切换

# 导入必要的模块
from startup.init_trainer import initialize_trainer # 用于根据配置初始化 Trainer
from config.training_params import TrainingConfig # 导入训练参数配置
from plot_utils import plot_training_progress   # 新增：导入公共绘图工具
import os # 操作系统相关功能，用于路径操作
import json # 用于读写 JSON 格式的日志
import torch # PyTorch 库，用于加载模型

# === 修改部分：设置路径 ===
# 原因：统一路径管理，保存日志和图表
PROJECT_ROOT = r"E:\ai_work\Flappy Bird" # 项目根目录，方便管理相对路径
LOG_DIR = os.path.join(PROJECT_ROOT, "logs") # 日志文件存放目录
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots") # 绘图文件存放目录
os.makedirs(LOG_DIR, exist_ok=True) # 如果目录不存在则创建
os.makedirs(PLOT_DIR, exist_ok=True) # 如果目录不存在则创建
# === 修改结束 ===

def main():
    # 初始化训练器
    config = TrainingConfig() # 创建一个训练配置实例
    config.algorithm = "ppo"  # 设置要使用的算法，这里明确指定了 PPO (可改为 "dqn" 切换)
    trainer = initialize_trainer(config) # 根据配置初始化对应的 Trainer 类实例 (如 PPOTrainer)

    # 加载保存点 (Load Checkpoint)
    # 尝试从指定路径加载之前训练的模型
    save_path = os.path.join(PROJECT_ROOT, "flappy_bird_models_20250610_184044", "model_step_1007616.pth")
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path) # 加载保存的 checkpoint 字典
        trainer.network.load_state_dict(checkpoint['network_state_dict']) # 加载网络参数
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 加载优化器状态
        trainer.episode_rewards = checkpoint['episode_rewards'] # 加载之前记录的奖励列表
        trainer.episode_lengths = checkpoint['episode_lengths'] # 加载之前记录的步数列表
        print(f"从 {save_path} 加载模型，步数: {checkpoint['timesteps']}")
    
    # === 修改部分：初始化日志 ===
    # 原因：保存训练数据，便于分析和绘图
    train_log_path = os.path.join(LOG_DIR, "train_log.json") # 训练日志文件的路径
    train_results = [] # 用于存储每个 episode 的奖励和步数
    # === 修改结束 ===
    
    # 开始训练
    try:
        # 调用 trainer 实例的 train 方法开始训练
        # total_timesteps 和 start_timesteps 从 config 中获取
        save_dir = trainer.train(total_timesteps=config.total_timesteps, start_timesteps=config.start_timesteps)
        print(f"\n训练完成! 模型保存在: {save_dir}")
        
        # === 修改部分：保存训练日志 ===
        # 原因：记录奖励和步数数据，以便后续分析
        train_results = [
            {"episode": i + 1, "reward": r, "steps": l} # 整理数据为字典列表
            for i, (r, l) in enumerate(zip(trainer.episode_rewards, trainer.episode_lengths))
        ]
        with open(train_log_path, "w") as f: # 将训练结果保存为 JSON 格式
            json.dump(train_results, f, indent=4) # indent=4 使 JSON 格式化输出，易读
        # === 修改结束 ===
        
        # === 修改部分：绘制训练进度 ===
        # 原因：优化绘图，与 training_progress.png 一致
        trainer.plot_training_progress() # 调用 trainer 内部的绘图方法 (会生成 training_progress.png)
        # 再次调用公共绘图工具，可能用于生成更精细或不同命名的图
        plot_training_progress(
            trainer.episode_rewards,
            trainer.episode_lengths,
            save_path=os.path.join(PLOT_DIR, "train_progress.png") # 保存到 plots 目录
        )
        print(f"训练进度图已保存至: {PLOT_DIR}\\train_progress.png")
        # === 修改结束 ===
        
    except KeyboardInterrupt:
        # 捕获用户中断 (Ctrl+C)，优雅地停止训练并保存进度
        print("\n训练被用户中断")
        trainer.plot_training_progress() # 绘制中断时的进度
        # 保存日志 (与正常完成时逻辑类似)
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
        # 捕获其他异常，打印错误信息和堆栈跟踪
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()