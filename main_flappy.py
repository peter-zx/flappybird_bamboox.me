# === 文件：Flappy Bird AI 主入口脚本 ===
# 位置：E:\ai_work\Flappy Bird\main_flappy.py

import os
import sys
from datetime import datetime

# 确保项目根目录在 sys.path 中，以便正确导入模块
# 当前文件所在目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = current_file_dir 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入自定义模块
from algorithms.ppo import PPOTrainer
# 导入环境注册模块，确保 gym.make('FlappyBird-v0') 可以找到环境
import flappy_bird_gymnasium # <--- 这一行很重要，确保环境被注册

# 定义训练和测试的配置类
class TrainingConfig:
    def __init__(self):
        self.env_name = 'FlappyBird-v0'
        self.n_envs = 16 # 并行环境数量
        self.total_timesteps = 100000 # 总训练步数
        self.lr = 2.5e-4 # 学习率
        self.gamma = 0.99 # 折扣因子
        self.gae_lambda = 0.95 # GAE lambda参数
        self.clip_coef = 0.1 # PPO裁剪系数
        self.vf_coef = 0.5 # 值函数损失系数
        self.ent_coef = 0.01 # 熵损失系数
        self.log_interval_episodes = 10 # 每隔多少个回合打印一次平均奖励
        self.save_interval = 10000 # 每隔多少总步数保存一次模型
        self.model_save_base_dir = os.path.join(PROJECT_ROOT, 'trained_models')
        self.plot_save_base_dir = os.path.join(PROJECT_ROOT, 'plots')
        self.load_model_path = None # 恢复训练时指定模型路径
        self.render_mode = None # 训练时不渲染
        self.test_render_mode = 'human' # 测试时渲染


class TestConfig(TrainingConfig):
    """用于测试模式的配置，继承自 TrainingConfig 并覆盖测试相关参数"""
    def __init__(self):
        super().__init__()
        self.n_envs = 1 # 测试时通常只用一个环境
        self.test_episodes_after_train = 5 # 测试时运行的回合数
        self.render_mode = None # 测试时不渲染 (通常测试时只看最终结果)
        self.test_render_mode = 'human' # 测试时渲染（用于可视化）

# 辅助函数：显示菜单
def display_menu():
    print("\n--- 请选择操作 ---")
    print("1. 从零开始训练 Flappy Bird AI")
    print("2. 从指定模型继续训练 Flappy Bird AI (恢复训练)")
    print("3. 测试指定模型 (Flappy Bird AI)")
    print("0. 退出")
    print("------------------")

# 辅助函数：获取最新模型路径
def get_latest_model_path(base_dir):
    all_sessions = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not all_sessions:
        return None

    # 按修改时间排序，获取最新的会话目录
    latest_session_dir = max(all_sessions, key=os.path.getmtime)
    
    # 在最新会话目录中查找最新的 .pth 模型文件
    model_files = [os.path.join(latest_session_dir, f) for f in os.listdir(latest_session_dir) if f.endswith('.pth')]
    if not model_files:
        return None
    
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def main():
    while True:
        display_menu()
        choice = input("请输入你的选择 (0-3): ").strip()

        if choice == '0':
            print("退出程序。")
            break # 退出主循环

        elif choice == '1': # 从零开始训练
            train_config = TrainingConfig()
            print("\n--- 启动从零开始训练 ---")
            print(f"模型将保存到基目录: {train_config.model_save_base_dir}")
            print(f"训练图和日志将保存到基目录: {train_config.plot_save_base_dir}")
            
            trainer = PPOTrainer(train_config)
            try:
                trainer.train()
                print("训练完成！")
            except Exception as e:
                print(f"\n训练过程中发生错误: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '2': # 从指定模型继续训练
            resume_config = TrainingConfig()
            print("\n--- 启动从指定模型继续训练 ---")
            
            load_path_input = input("请输入要恢复训练的模型完整路径 (留空则使用最新模型): ").strip()
            
            if not load_path_input:
                latest_model = get_latest_model_path(resume_config.model_save_base_dir)
                if latest_model:
                    resume_config.load_model_path = latest_model
                    print(f"将从最新模型 {latest_model} 恢复训练。")
                else:
                    print("未找到任何已训练模型，无法恢复训练。请先从零开始训练。")
                    continue # 返回主菜单
            else:
                resume_config.load_model_path = load_path_input
                print(f"将从模型 {load_path_input} 恢复训练。")

            trainer = PPOTrainer(resume_config)
            try:
                trainer.train()
                print("恢复训练完成！")
            except Exception as e:
                print(f"\n恢复训练过程中发生错误: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '3': # 测试指定模型
            print("\n--- 启动模型测试 ---")
            test_config = TestConfig() # 使用测试模式的配置

            test_model_path = input("请输入要测试的模型完整路径 (留空则测试最新模型): ").strip()

            if not test_model_path:
                latest_model = get_latest_model_path(test_config.model_save_base_dir)
                if latest_model:
                    test_model_path = latest_model
                    print(f"将测试最新模型: {latest_model}")
                else:
                    print("未找到任何已训练模型，无法进行测试。请先从零开始训练。")
                    continue # 返回主菜单
            else:
                print(f"将测试指定模型: {test_model_path}")

            # 为测试模式创建一个 PPOTrainer 实例
            # 注意：这里我们使用 TrainingConfig 的结构，只为满足 PPOTrainer 的构造函数，
            # 实际测试逻辑在 PPOTrainer.test_agent 中。
            # 确保传递给 PPOTrainer 的 config 对象具有 test_render_mode 属性。
            temp_trainer_config = TrainingConfig()
            temp_trainer_config.env_name = test_config.env_name
            temp_trainer_config.n_envs = test_config.n_envs # 测试时通常是1个环境
            temp_trainer_config.test_render_mode = test_config.test_render_mode # 传递渲染模式
            temp_trainer_config.model_save_base_dir = test_config.model_save_base_dir # 传递模型保存目录
            temp_trainer_config.plot_save_base_dir = test_config.plot_save_base_dir # 传递绘图保存目录
            temp_trainer_config.test_episodes_after_train = test_config.test_episodes_after_train # 传递测试回合数

            trainer = PPOTrainer(temp_trainer_config) 
            
            try:
                trainer.test_agent(test_model_path)
                print("模型测试完成！")
            except Exception as e:
                print(f"\n测试过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
            
            # 在测试完成后退出程序
            break # <--- 在这里添加 break 语句，跳出主循环

        else:
            print("无效的选择，请重新输入。")

if __name__ == "__main__":
    main()