# === 文件：算法核心 PPO 训练器 ===
# 位置：E:\ai_work\Flappy Bird\algorithms\ppo.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

# **新增：导入 gymnasium.vector 模块**
import gymnasium.vector 

# 确保项目根目录在 sys.path 中 (如果 ppo.py 被单独运行，确保路径正确)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_file_dir) 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithms.networks import PPONetwork
from algorithms.policy_update import ppo_update 
from algorithms.rollout import RolloutStorage 
import plot_utils 

# 辅助函数，确保目录存在
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化环境
        print(f"初始化环境 '{config.env_name}' (并行 {config.n_envs} 个)")
        
        # 修正：使用 gymnasium.vector.SyncVectorEnv 替代 gymnasium.vector.make
        # 创建一个环境工厂函数列表
        env_fns = [
            lambda: gym.make(config.env_name, render_mode=config.render_mode)
            for _ in range(config.n_envs)
        ]
        # 使用 SyncVectorEnv 包装这些环境
        self.envs = gymnasium.vector.SyncVectorEnv(env_fns)


        self.obs_dim = self.envs.single_observation_space.shape[0]
        self.action_dim = self.envs.single_action_space.n

        # 初始化网络
        self.network = PPONetwork(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr, eps=1e-5)
        
        # 初始化 RolloutStorage
        self.rollout_storage = RolloutStorage(
            self.obs_dim, 
            self.action_dim, 
            self.device, 
            num_steps=2048, # 每个环境的 rollout 步数
            num_envs=config.n_envs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda
        )

        # 训练日志和绘图数据
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_timesteps_elapsed = 0

        # 根据配置加载模型 (用于恢复训练)
        if self.config.load_model_path:
            self._load_model(self.config.load_model_path)
        else:
            print("未指定加载模型路径，将从头开始训练或测试。")
        
        # 确保模型和绘图保存目录存在 (在 main_flappy.py 中已经处理，这里作为保险)
        ensure_dir(self.config.model_save_base_dir)
        ensure_dir(self.config.plot_save_base_dir)


    def _load_model(self, model_path):
        """从指定路径加载模型权重。"""
        try:
            print(f"正在加载模型: {model_path}...")
            # 加载完整的 checkpoint，包含网络状态和优化器状态
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            # 只有当优化器状态存在时才加载，防止首次加载时没有
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 恢复训练步数
            if 'total_timesteps_elapsed' in checkpoint:
                self.total_timesteps_elapsed = checkpoint['total_timesteps_elapsed']
                print(f"模型已加载，恢复训练从 {self.total_timesteps_elapsed} 步开始。")
            if 'episode_rewards' in checkpoint:
                self.episode_rewards = checkpoint['episode_rewards']
            if 'episode_lengths' in checkpoint:
                self.episode_lengths = checkpoint['episode_lengths']
            print("模型加载成功。")
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise ValueError(f"加载模型失败: {e}")

    def _save_model(self, save_dir, plot_dir, current_timesteps, is_final=False): 
        """保存模型状态、训练日志和进度图。"""
        ensure_dir(save_dir) # 确保模型会话目录存在
        ensure_dir(plot_dir) # 确保绘图目录存在

        model_path = os.path.join(save_dir, f"model_step_{current_timesteps}.pth")
        
        # 将日志和图保存到 plot_dir
        log_filename = f"train_log_step_{current_timesteps}.json" if not is_final else "train_log.json"
        plot_filename = f"training_progress_step_{current_timesteps}.png" if not is_final else "training_progress.png"

        log_path = os.path.join(plot_dir, log_filename) 
        plot_path = os.path.join(plot_dir, plot_filename) 

        # 构造要保存的 checkpoint 数据
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps_elapsed': current_timesteps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'config': self.config.__dict__ # 保存配置，方便未来加载时查看
        }
        
        torch.save(checkpoint, model_path)
        
        # 保存训练进度图
        if len(self.episode_rewards) > 0:
            plot_utils.plot_training_progress(self.episode_rewards, self.episode_lengths, plot_path)
        
        # 保存训练日志 (确保 save_log 方法存在)
        self.save_log(log_path) 

        print(f"模型已保存到: {model_path}")
        print(f"训练进度图已保存到: {plot_path}")
        print(f"训练日志已保存到: {log_path}")


    def save_log(self, log_path):
        """保存训练日志到 JSON 文件。"""
        log_data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "total_timesteps": self.total_timesteps_elapsed
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)


    def train(self):
        """PPO 训练循环"""
        obs, info = self.envs.reset()
        self.rollout_storage.to_device(self.device) # 确保 rollouts 存储在正确设备上
        # 修正：RolloutStorage 初始插入只需观测和 mask
        # 确保初始观测也是 float32
        self.rollout_storage.observations[0].copy_(torch.from_numpy(obs).to(torch.float32))
        self.rollout_storage.masks[0].fill_(1.0) # 初始状态都是未结束


        total_steps_elapsed = self.total_timesteps_elapsed # 从加载的模型中恢复步数
        episodes_completed = len(self.episode_rewards) # 从加载的模型中恢复已完成的回合数
        last_log_episode_count = episodes_completed # 记录上次日志打印时的回合数

        # 获取当前的日期时间戳，用于创建唯一的训练会话目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # **模型会话目录的绝对路径，确保基于 config 中已经处理好的绝对路径**
        current_train_session_dir = os.path.join(
            self.config.model_save_base_dir, 
            f"flappy_bird_models_{timestamp}"
        )
        ensure_dir(current_train_session_dir) # 确保会话目录存在

        print(f"当前训练会话模型将保存到: {current_train_session_dir}")

        # 添加一个步数计数器，用于打印更频繁的进度信息
        step_counter_for_log = 0 
        log_steps_interval = 1000 # 每隔 1000 环境步打印一次基本进度

        # 训练循环
        while total_steps_elapsed < self.config.total_timesteps:
            # 1. 收集 Rollout 数据
            for step in range(self.rollout_storage.num_steps): 
                with torch.no_grad():
                    value, dist = self.network(self.rollout_storage.observations[step])
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                # 环境步进
                obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                done = terminated | truncated # 任何一个为 True 都表示回合结束

                # 记录回合奖励和长度
                for i, d in enumerate(done):
                    if d:
                        episodes_completed += 1
                        # 从 info['final_info'] 获取实际的最终奖励和长度
                        if info.get("_final_info") is not None and info["_final_info"][i] is not None:
                            self.episode_rewards.append(info["_final_info"][i]["episode_reward"])
                            self.episode_lengths.append(info["_final_info"][i]["episode_length"])
                        
                # 插入数据到 RolloutStorage
                self.rollout_storage.insert(
                    obs, action, reward, value, log_prob, done, info
                )
                total_steps_elapsed += self.config.n_envs # 总步数累加 (并行环境数)
                step_counter_for_log += self.config.n_envs # 更新日志步数计数器

                # 打印更频繁的进度信息
                if step_counter_for_log >= log_steps_interval:
                    current_average_reward = np.mean(self.episode_rewards[-self.config.log_interval_episodes:]) if len(self.episode_rewards) > 0 else 0
                    current_average_length = np.mean(self.episode_lengths[-self.config.log_interval_episodes:]) if len(self.episode_lengths) > 0 else 0
                    
                    print(f"Timesteps: {total_steps_elapsed}/{self.config.total_timesteps} "
                          f"(~{total_steps_elapsed / self.config.total_timesteps * 100:.2f}%) | "
                          f"Episodes: {episodes_completed} | "
                          f"Avg Reward (last {min(self.config.log_interval_episodes, len(self.episode_rewards))} episodes): {current_average_reward:.2f} | "
                          f"Avg Length: {current_average_length:.2f}")
                    step_counter_for_log = 0 # 重置计数器

            # 2. 计算返回和优势
            with torch.no_grad():
                next_value = self.network(self.rollout_storage.observations[-1])[0]
            self.rollout_storage.compute_returns(next_value)

            # 3. PPO 更新
            ppo_update(
                self.network,
                self.optimizer,
                self.rollout_storage, 
                self.config.clip_coef,
                self.config.vf_coef,
                self.config.ent_coef,
                num_mini_batch=self.rollout_storage.num_envs, # 每个环境作为一个 mini-batch
                update_epochs=4, # PPO 更新次数 (常用 4)
                device=self.device
            )
            self.rollout_storage.after_update() # 重置 rollout storage

            # 4. 保存模型和日志
            if (total_steps_elapsed % self.config.save_interval == 0 and total_steps_elapsed > 0) or \
               (total_steps_elapsed >= self.config.total_timesteps):
                self._save_model(current_train_session_dir, self.config.plot_save_base_dir, total_steps_elapsed) 
                if total_steps_elapsed >= self.config.total_timesteps:
                    break # 达到总步数后退出循环

        # 训练结束时的最终保存和测试
        self._save_model(current_train_session_dir, self.config.plot_save_base_dir, total_steps_elapsed, is_final=True) 
        
        print("\n训练完成！")
        self.envs.close() # 关闭环境
        
        return current_train_session_dir # 返回模型保存的会话目录


    def test_agent(self, model_path):
        """测试训练后的代理"""
        print(f"\n--- 开始测试代理 (加载模型: {model_path}) ---")
        
        test_env = gym.make(self.config.env_name, render_mode=self.config.test_render_mode)
        
        test_network = PPONetwork(self.obs_dim, self.action_dim).to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False) 
            test_network.load_state_dict(checkpoint['network_state_dict'])
            test_network.eval() 
            print("测试模型加载成功。")
        except Exception as e:
            print(f"测试模型加载失败: {e}")
            test_env.close()
            return

        total_test_rewards = []
        total_test_lengths = []

        for episode in range(self.config.test_episodes_after_train):
            obs, info = test_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                with torch.no_grad():
                    # 确保测试时的观测也是 float32
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    _, dist = test_network(obs_tensor)
                    action = dist.sample().item() # 从策略分布中采样动作

                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                if done:
                    total_test_rewards.append(episode_reward)
                    total_test_lengths.append(episode_length)
                    print(f"测试回合 {episode + 1}: 奖励 = {episode_reward:.2f}, 长度 = {episode_length}")
                    
        test_env.close()
        
        if total_test_rewards:
            avg_test_reward = np.mean(total_test_rewards)
            avg_test_length = np.mean(total_test_lengths)
            print(f"\n测试完成！平均奖励: {avg_test_reward:.2f}, 平均长度: {avg_test_length:.2f}")

        print("--- 测试结束 ---")