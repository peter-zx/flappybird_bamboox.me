# === 文件：训练主逻辑 ===
# 位置：E:\ai_work\Flappy Bird\algorithms\trainer.py
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import os
from .networks import PPONetwork
from .rollout import collect_rollouts
from .policy_update import update_policy

class PPOTrainer:
    def __init__(self, env_name='FlappyBird-v0', lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, n_envs=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        try:
            from gymnasium.vector import SyncVectorEnv
            self.env = SyncVectorEnv([lambda: gym.make(env_name, render_mode="rgb_array") for _ in range(n_envs)])
            obs = self.env.reset()[0]
            self.obs_dim = obs.shape[1] if len(obs.shape) > 1 else obs.shape[0]
        except ImportError:
            print("向量环境支持缺失，降级为单环境")
            self.env = gym.make(env_name, render_mode="rgb_array")
            obs = self.env.reset()
            self.obs_dim = obs.shape[0]
        
        # === 强制动作维度 ===
        self.action_dim = 8  # 匹配模型 model_step_1007616.pth
        print(f"强制动作维度：{self.action_dim}")
        # === 结束 ===
        
        self.network = PPONetwork(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.episode_rewards = []
        self.episode_lengths = []
        self.n_envs = n_envs
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        
        for t in range(len(rewards)):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * (advantages[t + 1] if t + 1 < len(rewards) else 0)
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def train(self, total_timesteps=1000000, save_interval=100000, start_timesteps=0):
        print("开始训练 Flappy Bird AI...")
        print(f"目标训练步数: {total_timesteps}, 从 {start_timesteps} 开始")
        print(f"观察空间维度: {self.obs_dim}")
        print(f"动作空间大小: {self.action_dim}")
        print(f"并行环境数量: {self.n_envs}")
        
        timesteps = start_timesteps
        update_count = start_timesteps // (2048 * self.n_envs)
        
        save_dir = f"flappy_bird_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)
        
        while timesteps < total_timesteps:
            print(f"\n=== 更新 {update_count + 1} ===")
            print(f"已训练步数: {timesteps}/{total_timesteps}")
            
            batch_data = collect_rollouts(self.env, self.network, self.device, self.n_envs)
            timesteps += len(batch_data['rewards']) * self.n_envs
            
            with torch.no_grad():
                last_obs = torch.FloatTensor(batch_data['observations'][-1]).to(self.device)
                _, _, _, next_value = self.network.get_action_and_value(last_obs)
                next_value = next_value.cpu().numpy().squeeze()
            
            advantages, returns = self.compute_gae(
                batch_data['rewards'],
                batch_data['values'],
                batch_data['dones'],
                next_value
            )
            
            batch_data['advantages'] = advantages
            batch_data['returns'] = returns
            
            update_policy(self.network, self.optimizer, batch_data, self.device, self.clip_coef, self.vf_coef, self.ent_coef)
            update_count += 1
            
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10 * self.n_envs:]
                avg_reward = np.mean(recent_rewards)
                max_reward = np.max(recent_rewards)
                print(f"最近10个episode平均奖励: {avg_reward:.2f}")
                print(f"最近10个episode最高奖励: {max_reward:.2f}")
                print(f"总完成episode数: {len(self.episode_rewards)}")
            
            if timesteps % save_interval == 0 or timesteps >= total_timesteps:
                model_path = os.path.join(save_dir, f"model_step_{timesteps}.pth")
                torch.save({
                    'network_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'timesteps': timesteps,
                    'episode_rewards': self.episode_rewards,
                    'episode_lengths': self.episode_lengths
                }, model_path)
                print(f"模型已保存: {model_path}")
        
        print("\n训练完成!")
        
        test_env = gym.make('FlappyBird-v0', render_mode="human")
        obs, _ = test_env.reset()
        total_reward = 0
        for _ in range(1000):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                action, _, _, _ = self.network.get_action_and_value(obs_tensor)
                # === 动作映射 ===
                action = action.item() % 2  # 映射到 [0, 1]（Discrete(2)）
                # === 结束 ===
            obs, reward, done, truncated, _ = test_env.step(action)
            total_reward += reward
            test_env.render()
            if done or truncated:
                print(f"Game Over! Total Reward: {total_reward}")
                total_reward = 0
                obs, _ = test_env.reset()
        test_env.close()
        
        return save_dir
    
    def plot_training_progress(self):
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
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def select_action(self, obs_tensor):
        """Select action for testing"""
        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(obs_tensor)
            # === 动作映射 ===
            action = action.item() % 2  # 映射到 [0, 1]（Discrete(2)）
            # === 结束 ===
            return action