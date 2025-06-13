# === 文件：PPO 算法实现 ===
# 位置：E:\ai_work\Flappy Bird\algorithms\ppo.py
# 用途：包含 PPOTrainer 类，负责 PPO 算法的训练流程和测试

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv 
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json # 用于保存训练日志
import pygame # 用于测试时的 human 渲染

# 从当前目录导入网络、rollout 和策略更新模块
from .networks import PPONetwork 
from .rollout import collect_rollouts
from .policy_update import update_policy

class PPOTrainer:
    def __init__(self, config): # 接受配置对象
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 环境初始化 (训练环境)
        try:
            self.env = SyncVectorEnv([
                lambda: gym.make(self.config.env_name, render_mode=self.config.render_mode) 
                for _ in range(self.config.n_envs)
            ])
            # 获取观察空间和动作空间维度
            # 对于向量化环境，obs 是 (n_envs, obs_dim)
            obs_sample = self.env.reset()[0] 
            # 这里的 obs_sample 可能是 (n_envs, obs_dim) 或者 (obs_dim,) 如果 n_envs=1
            # 确保 obs_dim 的获取是正确的
            if len(obs_sample.shape) == 2: # (n_envs, obs_dim)
                self.obs_dim = obs_sample.shape[1]
            elif len(obs_sample.shape) == 1: # (obs_dim,) for single env reset
                self.obs_dim = obs_sample.shape[0]
            else: # Fallback for unexpected shapes
                raise ValueError(f"无法确定观察空间维度。obs_sample.shape: {obs_sample.shape}")

            self.action_dim = self.env.single_action_space.n # 从环境中动态获取动作维度
        except Exception as e:
            print(f"初始化并行环境失败: {e}，尝试初始化单个环境。")
            # 降级为单个环境
            self.env = gym.make(self.config.env_name, render_mode=self.config.render_mode)
            obs_sample, _ = self.env.reset()
            self.obs_dim = obs_sample.shape[0]
            self.action_dim = self.env.action_space.n
            self.config.n_envs = 1 # 更新并行环境数量为 1

        print(f"检测到的观察空间维度: {self.obs_dim}")
        print(f"检测到的动作空间大小: {self.action_dim}")
        print(f"并行环境数量: {self.config.n_envs}")
        
        # 初始化 PPO 网络
        self.network = PPONetwork(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)
        
        # 初始化训练统计数据
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_timesteps_elapsed = 0 # 实际训练中累积的总步数
        self.current_episode_count = 0 # 记录完成的 episode 数量
        self.log_data = [] # 用于保存 JSON 格式的日志

        # 加载预训练模型 (如果指定了路径)
        if self.config.load_model_path:
            self._load_model(self.config.load_model_path)
        else:
            print("未指定加载模型路径，将从头开始训练。")
        
    def _load_model(self, path):
        """加载模型和优化器状态"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            # 尝试加载优化器状态，如果不存在则跳过 (为了兼容性)
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.total_timesteps_elapsed = checkpoint['timesteps']
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_lengths = checkpoint.get('episode_lengths', [])
            self.current_episode_count = len(self.episode_rewards) # 恢复 episode 计数
            print(f"成功从 {path} 恢复模型。已训练步数: {self.total_timesteps_elapsed}")
            print(f"已完成 episode 数: {self.current_episode_count}")
            # 设置从保存点继续训练
            self.config.start_timesteps = self.total_timesteps_elapsed
        except FileNotFoundError:
            print(f"错误: 找不到模型文件 {path}。")
            if self.config.algorithm == "ppo" and self.config.load_model_path:
                raise FileNotFoundError(f"加载模型失败: 找不到文件 {path}")
            self.total_timesteps_elapsed = 0 
        except Exception as e:
            print(f"加载模型时发生错误: {e}。")
            if self.config.algorithm == "ppo" and self.config.load_model_path:
                raise ValueError(f"加载模型失败: {e}")
            self.total_timesteps_elapsed = 0 

    def _save_model(self, save_path):
        """保存模型和训练状态"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timesteps': self.total_timesteps_elapsed,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, save_path)
        print(f"模型已保存: {save_path}")

    def compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计 (GAE) 和回报"""
        # next_value 已经是 (n_envs,) 的 numpy 数组
        # 如果 n_envs 为 1，它会是 (1,)，这正是我们想要的
        
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        
        # 确保 next_value 形状正确 (n_envs,)
        # 如果 next_value 是标量 (当 n_envs=1 且 .squeeze() 导致)，需要将其转为 (1,)
        # 这里使用 np.atleast_1d 确保它至少是一维数组
        next_value = np.atleast_1d(next_value) 
        
        # 从后往前计算 GAE
        last_gae_lam = np.zeros(self.config.n_envs) # GAE 的初始值是 0 for each env
        
        # 遍历时间步，注意这里是从最后一个时间步开始倒序遍历
        for t in reversed(range(rewards.shape[0])): 
            next_non_terminal = 1.0 - dones[t] # (n_envs,)，如果 episode 结束则为 0

            if t == rewards.shape[0] - 1: # 如果是 rollout 的最后一个时间步
                current_next_value = next_value # 使用传入的 next_value (来自最后一个 obs)
            else: # 否则，使用当前批次中下一个时间步的值函数估计
                current_next_value = values[t + 1] 
            
            delta = rewards[t] + self.config.gamma * current_next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values 
        return advantages, returns
    
    def train(self):
        print("\n--- 开始训练 Flappy Bird AI ---")
        print(f"目标训练步数: {self.config.total_timesteps}, 从 {self.total_timesteps_elapsed} 开始")
        
        current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
        PROJECT_ROOT = os.path.dirname(os.path.dirname(current_file_dir)) 
        
        base_model_save_path = os.path.join(PROJECT_ROOT, self.config.model_save_base_dir)
        os.makedirs(base_model_save_path, exist_ok=True)

        current_train_session_dir = os.path.join(base_model_save_path, f"flappy_bird_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(current_train_session_dir, exist_ok=True)
        print(f"本次训练的模型将保存到: {current_train_session_dir}")

        current_timesteps = self.total_timesteps_elapsed # 从加载的步数或 0 开始
        
        # 获取环境初始状态
        obs_np, _ = self.env.reset()
        if self.config.n_envs == 1 and len(obs_np.shape) == 1:
            obs_np = np.expand_dims(obs_np, axis=0) # 确保是 (1, obs_dim) 

        while current_timesteps < self.config.total_timesteps:
            # 收集数据 (rollout)
            batch_data = collect_rollouts(self.env, self.network, self.device, self.config.n_envs, num_steps_per_rollout=2048)
            
            # 更新总步数
            current_timesteps += batch_data['num_timesteps_collected']
            self.total_timesteps_elapsed = current_timesteps

            # 获取下一个状态的价值，用于 GAE 计算
            # batch_data['next_observations'] 现在已经是 (n_envs, obs_dim) 的 numpy 数组
            with torch.no_grad():
                last_obs_tensor = torch.FloatTensor(batch_data['next_observations']).to(self.device) 
                _, _, _, next_value_tensor = self.network.get_action_and_value(last_obs_tensor)
                # 确保 next_value_np 始终是 (n_envs,) 的 NumPy 数组
                next_value_np = next_value_tensor.cpu().numpy().flatten() # 使用 .flatten() 确保 (N,) 形状
            
            # 计算优势和回报
            advantages, returns = self.compute_gae(
                batch_data['rewards'],
                batch_data['values'],
                batch_data['dones'],
                next_value_np 
            )
            
            batch_data['advantages'] = advantages
            batch_data['returns'] = returns
            
            # 更新策略和值函数
            loss_info = update_policy(self.network, self.optimizer, batch_data, 
                                    self.device, self.config.clip_coef, 
                                    self.config.vf_coef, self.config.ent_coef)
            
            # 记录 episode 奖励和长度 (从 collect_rollouts 收集到的)
            if 'episode_rewards_in_rollout' in batch_data and 'episode_lengths_in_rollout' in batch_data:
                self.episode_rewards.extend(batch_data['episode_rewards_in_rollout'])
                self.episode_lengths.extend(batch_data['episode_lengths_in_rollout'])
                self.current_episode_count = len(self.episode_rewards) 

            # 打印训练日志
            if self.current_episode_count >= self.config.log_interval_episodes and \
            self.current_episode_count % self.config.log_interval_episodes < self.config.n_envs: 
                recent_rewards = self.episode_rewards[-self.config.log_interval_episodes:]
                avg_reward = np.mean(recent_rewards)
                max_reward = np.max(recent_rewards)
                
                print(f"\n--- 训练日志 (步数: {current_timesteps}/{self.config.total_timesteps}) ---")
                print(f"平均策略损失: {loss_info['policy_loss']:.4f}")
                print(f"平均值函数损失: {loss_info['value_loss']:.4f}")
                print(f"平均熵损失: {loss_info['entropy_loss']:.4f}")
                print(f"最近 {len(recent_rewards)} 个 episode 平均奖励: {avg_reward:.2f}")
                print(f"最近 {len(recent_rewards)} 个 episode 最高奖励: {max_reward:.2f}")
                print(f"总完成 episode 数: {self.current_episode_count}")

                self.log_data.append({
                    'timesteps': current_timesteps,
                    'episode_count': self.current_episode_count,
                    'avg_reward': avg_reward,
                    'max_reward': max_reward,
                    'policy_loss': loss_info['policy_loss'],
                    'value_loss': loss_info['value_loss'],
                    'entropy_loss': loss_info['entropy_loss'],
                    'timestamp': datetime.now().isoformat()
                })
            
            # 保存模型
            if (current_timesteps // self.config.save_interval) > ((current_timesteps - batch_data['num_timesteps_collected']) // self.config.save_interval):
                model_path = os.path.join(current_train_session_dir, f"model_step_{current_timesteps}.pth")
                self._save_model(model_path)
        
        # 训练完成，保存最终模型
        final_model_path = os.path.join(current_train_session_dir, f"model_step_{self.config.total_timesteps}.pth")
        self._save_model(final_model_path)
        print("\n训练完成!")

        log_file_path = os.path.join(current_train_session_dir, "train_log.json")
        with open(log_file_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)
        print(f"训练日志已保存到: {log_file_path}")

        self.plot_training_progress(current_train_session_dir)

        self.test_agent(final_model_path)
        
        return current_train_session_dir 

    def plot_training_progress(self, save_dir):
        """绘制训练进度图并保存"""
        if len(self.episode_rewards) == 0:
            print("没有训练数据可以绘制")
            return
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close() 
        print(f"训练进度图已保存到: {plot_path}")

    def test_agent(self, model_path):
        """测试训练后的代理"""
        print(f"\n--- 开始测试代理 (加载模型: {model_path}) ---")
        
        test_env = gym.make(self.config.env_name, render_mode=self.config.test_render_mode)
        
        test_network = PPONetwork(self.obs_dim, self.action_dim).to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False) # <-- 修改这一行！
            test_network.load_state_dict(checkpoint['network_state_dict'])
            test_network.eval() 
            print("测试模型加载成功。")
        except Exception as e:
            print(f"测试模型加载失败: {e}")
            test_env.close()
            return

        test_episode_rewards = []
        for i in range(self.config.test_episodes_after_train):
            obs, _ = test_env.reset()
            current_episode_reward = 0
            done = False
            truncated = False
            steps = 0
            while not done and not truncated:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0) 
                    action, _, _, _ = test_network.get_action_and_value(obs_tensor)
                    action = action.item() 
                
                obs, reward, done, truncated, _ = test_env.step(action)
                current_episode_reward += reward
                steps += 1
                
                if self.config.test_render_mode == "human":
                    test_env.render()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            test_env.close()
                            print("测试被用户中断。")
                            return
            test_episode_rewards.append(current_episode_reward)
            print(f"测试回合 {i+1}: 总奖励: {current_episode_reward:.2f}, 步数: {steps}")
        
        test_env.close()
        print("--- 测试结束 ---")
        if test_episode_rewards:
            avg_test_reward = np.mean(test_episode_rewards)
            print(f"所有测试回合的平均奖励: {avg_test_reward:.2f}")

    def select_action(self, obs_tensor):
        """用于在训练或测试中选择动作 (单一环境的观测)"""
        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(obs_tensor)
            return action.item()