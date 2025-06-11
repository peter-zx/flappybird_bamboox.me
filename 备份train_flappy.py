import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 手动注册环境
register(
    id='FlappyBird-v0',
    entry_point='flappy_bird_gymnasium.envs:FlappyBirdEnv',
)

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # 策略网络 (Actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, action_dim)
        )
        
        # 价值网络 (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        policy_logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return policy_logits, value
    
    def get_action_and_value(self, x, action=None):
        policy_logits, value = self.forward(x)
        probs = Categorical(logits=policy_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value

class PPOTrainer:
    def __init__(self, env_name='FlappyBird-v0', lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_coef=0.2, vf_coef=0.5, ent_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        self.network = PPONetwork(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # PPO 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        
    def collect_rollouts(self, num_steps=2048):
        """收集经验数据"""
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(obs.unsqueeze(0))
            
            observations.append(obs.cpu().numpy())
            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            values.append(value.cpu().numpy())
            
            # 执行动作
            next_obs, reward, done, truncated, _ = self.env.step(action.item())
            rewards.append(reward)
            dones.append(done or truncated)
            
            episode_reward += reward
            episode_length += 1
            
            if done or truncated:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                print(f"Episode 完成! 奖励: {episode_reward:.2f}, 长度: {episode_length}")
                
                obs, _ = self.env.reset()
                obs = torch.FloatTensor(obs).to(self.device)
                episode_reward = 0
                episode_length = 0
            else:
                obs = torch.FloatTensor(next_obs).to(self.device)
        
        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values).flatten(),
            'log_probs': np.array(log_probs).flatten()
        }
    
    def compute_gae(self, rewards, values, dones, next_value):
        """计算 Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self, batch_data, num_epochs=4, batch_size=64):
        """更新策略网络"""
        observations = torch.FloatTensor(batch_data['observations']).to(self.device)
        actions = torch.LongTensor(batch_data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch_data['log_probs']).to(self.device)
        advantages = torch.FloatTensor(batch_data['advantages']).to(self.device)
        returns = torch.FloatTensor(batch_data['returns']).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(num_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(observations))
            
            for start in range(0, len(observations), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 获取新的策略输出
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO 裁剪损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # 更新网络
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
    
    def train(self, total_timesteps=1000000, save_interval=100000):
        """开始训练"""
        print("开始训练 Flappy Bird AI...")
        print(f"目标训练步数: {total_timesteps}")
        print(f"观察空间维度: {self.obs_dim}")
        print(f"动作空间大小: {self.action_dim}")
        
        timesteps = 0
        update_count = 0
        
        # 创建保存目录
        save_dir = f"flappy_bird_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)
        
        while timesteps < total_timesteps:
            # 收集经验
            print(f"\n=== 更新 {update_count + 1} ===")
            print(f"已训练步数: {timesteps}/{total_timesteps}")
            
            batch_data = self.collect_rollouts(num_steps=2048)
            timesteps += len(batch_data['rewards'])
            
            # 计算最后一个状态的价值（用于 GAE）
            with torch.no_grad():
                last_obs = torch.FloatTensor(batch_data['observations'][-1]).to(self.device)
                _, _, _, next_value = self.network.get_action_and_value(last_obs.unsqueeze(0))
                next_value = next_value.cpu().numpy().item()
            
            # 计算优势和回报
            advantages, returns = self.compute_gae(
                batch_data['rewards'], 
                batch_data['values'], 
                batch_data['dones'], 
                next_value
            )
            
            batch_data['advantages'] = advantages
            batch_data['returns'] = returns
            
            # 更新策略
            self.update_policy(batch_data)
            update_count += 1
            
            # 打印统计信息
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:]  # 最近10个episode
                avg_reward = np.mean(recent_rewards)
                max_reward = np.max(recent_rewards)
                print(f"最近10个episode平均奖励: {avg_reward:.2f}")
                print(f"最近10个episode最高奖励: {max_reward:.2f}")
                print(f"总完成episode数: {len(self.episode_rewards)}")
            
            # 保存模型
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
        return save_dir
    
    def plot_training_progress(self):
        """绘制训练进度"""
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

def main():
    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        print(f"CUDA 可用! 设备: {torch.cuda.get_device_name()}")
        print(f"CUDA 版本: {torch.version.cuda}")
    else:
        print("CUDA 不可用，使用 CPU 训练")
    
    # 创建训练器
    trainer = PPOTrainer(
        env_name='FlappyBird-v0',
        lr=3e-4,           # 学习率
        gamma=0.99,        # 折扣因子
        gae_lambda=0.95,   # GAE lambda
        clip_coef=0.2,     # PPO 裁剪系数
        vf_coef=0.5,       # 价值损失系数
        ent_coef=0.01      # 熵损失系数
    )
    
    # 开始训练
    try:
        save_dir = trainer.train(total_timesteps=500000)  # 50万步训练
        print(f"\n训练完成! 模型保存在: {save_dir}")
        
        # 绘制训练进度
        trainer.plot_training_progress()
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        trainer.plot_training_progress()
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()