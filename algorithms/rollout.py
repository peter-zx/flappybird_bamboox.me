# === 文件：Rollout 存储 ===
# 位置：E:\ai_work\Flappy Bird\algorithms\rollout.py

import torch
import numpy as np
import gymnasium as gym # 确保导入 gymnasium

class RolloutStorage:
    def __init__(self, obs_dim, action_dim, device, num_steps, num_envs, gamma, gae_lambda):
        # 确保 observations 是 float32 类型
        self.observations = torch.zeros(num_steps + 1, num_envs, obs_dim, dtype=torch.float32).to(device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, dtype=torch.float32).to(device)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1, dtype=torch.float32).to(device)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1, dtype=torch.float32).to(device)
        self.actions = torch.zeros(num_steps, num_envs, 1, dtype=torch.long).to(device) # Actions are discrete, use long
        self.log_probs = torch.zeros(num_steps, num_envs, 1, dtype=torch.float32).to(device)
        self.masks = torch.ones(num_steps + 1, num_envs, 1, dtype=torch.float32).to(device) # Mask for done episodes
        self.device = device
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.step = 0 # Current step in the rollout buffer

    def to_device(self, device):
        self.observations = self.observations.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.log_probs = self.log_probs.to(device)
        self.masks = self.masks.to(device)
        self.device = device

    def insert(self, obs, action, reward, value_pred, log_prob, done, info):
        # 处理 Gymnasium 返回的 numpy 数组，转换为 torch tensor
        # obs: (num_envs, obs_dim) -> self.observations[self.step + 1]: (num_envs, obs_dim)
        # 确保从 numpy 转换为 tensor 时是 float32
        self.observations[self.step + 1].copy_(torch.from_numpy(obs).to(torch.float32))
        # action: (num_envs,) -> action.unsqueeze(-1): (num_envs, 1)
        self.actions[self.step].copy_(action.unsqueeze(-1)) 
        # reward: (num_envs,) -> reward.unsqueeze(-1): (num_envs, 1)
        # 确保从 numpy 转换为 tensor 时是 float32
        self.rewards[self.step].copy_(torch.from_numpy(reward).unsqueeze(-1).to(torch.float32)) 
        # value_pred: (num_envs, 1)
        self.value_preds[self.step].copy_(value_pred)
        # log_prob: (num_envs,) -> log_prob.unsqueeze(-1): (num_envs, 1)
        self.log_probs[self.step].copy_(log_prob.unsqueeze(-1)) 
        
        # Mask: 1 if not done, 0 if done
        # done 是布尔数组 (num_envs,)，需要转换为 (num_envs, 1) 的 float32 Tensor
        self.masks[self.step + 1].copy_(torch.from_numpy(1.0 - done).unsqueeze(-1).to(torch.float32))

        self.step = (self.step + 1) % self.num_steps # Advance step, loop back if buffer full

        # 当 buffer 满了或者需要计算返回时，清空 buffer 并重新开始
        if self.step == 0:
            pass # buffer 满了，compute_returns 会处理，这里不需要额外操作

    def compute_returns(self, next_value):
        """
        计算 GAE (Generalized Advantage Estimation) 和 returns。
        """
        # A_t = delta_t + gamma * lambda * A_{t+1}
        # V_t = R_t + gamma * V_{t+1}
        
        # 将最后一个 value_pred 设置为下一个状态的值
        self.value_preds[-1].copy_(next_value)

        gae = 0
        for step in reversed(range(self.num_steps)):
            delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step] # R_t = A_t + V_t

    def after_update(self):
        """
        更新完成后，将最后一个观测值复制到第一个位置，并重置步数。
        """
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])
        self.step = 0 # Reset step to 0 for next rollout