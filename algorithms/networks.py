# === 文件：神经网络模型定义 ===
# 位置：E:\ai_work\Flappy Bird\algorithms\networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions # 导入 distributions 模块
import numpy as np # **新增：导入 numpy 模块**

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PPONetwork, self).__init__()

        # 定义特征提取器/公共网络部分
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # 定义 Actor (策略网络)
        # Actor 输出的是动作的 logits (未归一化的对数概率)，然后通过 Softmax 转换为概率分布
        self.actor = nn.Linear(128, action_dim)

        # 定义 Critic (值函数网络)
        # Critic 输出的是状态的值 (单个浮点数)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        # 如果输入是 NumPy 数组，将其转换为 PyTorch Tensor
        # 注意：在训练循环中通常会确保输入是 float32，这里主要用于测试时的鲁棒性
        if isinstance(x, np.ndarray): # 这里需要 np
            x = torch.tensor(x, dtype=torch.float32)
        
        # 确保输入是浮点型，如果它不是，并且需要梯度
        if x.dtype != torch.float32:
            x = x.float()

        # 确保输入维度正确 (batch_size, obs_dim)
        # 如果输入是 (obs_dim,)，需要添加 batch 维度
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 通过特征提取器
        features = self.feature_extractor(x)

        # 计算 Actor 的输出 (动作的 logits)
        action_logits = self.actor(features)
        
        # **关键修正**：创建 Categorical 分布对象
        # PPO 需要从这个分布中采样动作并计算 log_prob
        dist = distributions.Categorical(logits=action_logits)

        # 计算 Critic 的输出 (状态值)
        value = self.critic(features)

        # 返回值和分布对象
        return value, dist