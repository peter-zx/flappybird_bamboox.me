# === 文件：Flappy Bird AI 主入口脚本 ===
# 位置：E:\ai_work\Flappy Bird\main_flappy.py
# 用途：提供交互式菜单，选择训练（从零/恢复）或测试功能

import os
import sys
import glob
import torch
from datetime import datetime

# **新增这一行：导入 flappy_bird_gymnasium 库以注册环境**
import flappy_bird_gymnasium 

# 确保项目根目录在 sys.path 中，以便正确导入模块
current_file_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = current_file_dir 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithms.ppo import PPOTrainer
from config.training_params import TrainingConfig
from config.test_params import TestConfig

def get_latest_model_path(base_dir):
    """
    辅助函数：从指定的基础目录中找到最新的模型文件路径。
    """
    all_session_dirs = sorted(glob.glob(os.path.join(base_dir, 'flappy_bird_models_*')), reverse=True)
    if not all_session_dirs:
        return None, "没有找到训练会话目录。"

    latest_session_dir = all_session_dirs[0]
    # glob 匹配 model_step_*.pth，按修改时间倒序排列，以获取最新的
    model_files = sorted(glob.glob(os.path.join(latest_session_dir, 'model_step_*.pth')), key=os.path.getmtime, reverse=True)
    
    if not model_files:
        return None, f"在最新模型目录 '{latest_session_dir}' 中未找到任何模型文件。"
    
    return model_files[0], None # 返回找到的最新模型路径和 None (表示无错误)

def display_menu():
    """显示操作菜单"""
    print("\n--- 请选择操作 ---")
    print("1. 从零开始训练 Flappy Bird AI")
    print("2. 从指定模型继续训练 Flappy Bird AI (恢复训练)")
    print("3. 测试指定模型 (Flappy Bird AI)")
    print("0. 退出")
    print("------------------")

def main():
    while True:
        display_menu()
        choice = input("请输入你的选择 (0-3): ").strip()

        if choice == '0':
            print("退出程序。")
            break
        
        elif choice == '1': # 从零开始训练
            print("\n--- 启动从零开始训练 ---")
            train_config = TrainingConfig() 
            train_config.load_model_path = None # 明确设置为 None
            
            # 将相对路径转换为绝对路径，并更新到配置中
            train_config.model_save_base_dir = os.path.join(PROJECT_ROOT, train_config.model_save_base_dir)
            train_config.plot_save_base_dir = os.path.join(PROJECT_ROOT, train_config.plot_save_base_dir)
            
            # 确保 plots 目录存在
            os.makedirs(train_config.plot_save_base_dir, exist_ok=True)

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
            print("\n--- 启动恢复训练 ---")
            model_path_to_resume = input("请输入要恢复训练的模型完整路径 (例如: E:\\path\\to\\model.pth): ").strip()
            
            if not os.path.exists(model_path_to_resume):
                print(f"错误: 模型文件不存在于路径: {model_path_to_resume}")
                continue

            train_config = TrainingConfig() 
            train_config.load_model_path = model_path_to_resume # 动态设置加载路径

            # 将相对路径转换为绝对路径，并更新到配置中
            train_config.model_save_base_dir = os.path.join(PROJECT_ROOT, train_config.model_save_base_dir)
            train_config.plot_save_base_dir = os.path.join(PROJECT_ROOT, train_config.plot_save_base_dir)

            # 确保 plots 目录存在
            os.makedirs(train_config.plot_save_base_dir, exist_ok=True)
            
            print(f"模型将保存到基目录: {train_config.model_save_base_dir}")
            print(f"训练图和日志将保存到基目录: {train_config.plot_save_base_dir}")

            trainer = PPOTrainer(train_config)
            try:
                print(f"将从模型 {model_path_to_resume} 恢复训练。")
                trainer.train()
                print("恢复训练完成！")
            except Exception as e:
                print(f"\n恢复训练过程中发生错误: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '3': # 测试指定模型
            print("\n--- 启动模型测试 ---")
            test_config = TestConfig() 

            test_model_path = input("请输入要测试的模型完整路径 (留空则测试最新模型): ").strip()

            # 将相对路径转换为绝对路径，并更新到配置中
            test_config.model_search_base_dir = os.path.join(PROJECT_ROOT, test_config.model_search_base_dir)
            test_config.plot_save_base_dir = os.path.join(PROJECT_ROOT, test_config.plot_save_base_dir) # 尽管测试通常不保存新图，但保持一致性

            # 确保 plots 目录存在
            os.makedirs(test_config.plot_save_base_dir, exist_ok=True)

            if test_model_path == "": # 用户留空，自动查找最新
                model_path_from_latest, error_msg = get_latest_model_path(test_config.model_search_base_dir) # 使用修改后的 search_base_dir
                if model_path_from_latest:
                    test_model_path = model_path_from_latest
                    print(f"将测试最新模型: {test_model_path}")
                else:
                    print(f"错误: {error_msg}")
                    continue 
            elif not os.path.exists(test_model_path):
                print(f"错误: 模型文件不存在于路径: {test_model_path}")
                continue
            
            # 为 PPOTrainer 创建一个兼容的配置对象
            temp_trainer_config = TrainingConfig() # 使用 TrainingConfig 的结构，只为满足 PPOTrainer 的构造函数
            temp_trainer_config.env_name = test_config.env_name
            temp_trainer_config.n_envs = test_config.n_envs 
            temp_trainer_config.test_render_mode = test_config.test_render_mode
            temp_trainer_config.test_episodes_after_train = test_config.test_episodes_after_train
            temp_trainer_config.load_model_path = test_model_path 
            
            # **重要：将 plot_save_base_dir 传递给 temp_trainer_config，因为测试模式也可能需要保存图 (例如测试结果图)**
            temp_trainer_config.plot_save_base_dir = test_config.plot_save_base_dir 
            temp_trainer_config.model_save_base_dir = test_config.model_search_base_dir # 这个在测试时实际不用于保存，但为了结构完整性

            trainer = PPOTrainer(temp_trainer_config) 
            
            try:
                trainer.test_agent(test_model_path)
                print("模型测试完成！")
            except Exception as e:
                print(f"\n测试过程中发生错误: {e}")
                import traceback
                traceback.print_exc()

        else:
            print("无效的选择，请重新输入。")

if __name__ == "__main__":
    main()