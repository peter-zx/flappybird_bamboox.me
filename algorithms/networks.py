# === 文件：网络定义 ===
# 位置：E:\ai_work\Flappy Bird\algorithms\networks.py
# 用途：定义 PPO 算法中使用的神经网络结构

import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        
        # 将 obs_dim 保存为类的属性，以便在其他地方访问
        self.obs_dim = obs_dim # <-- **增加这一行！**
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim), # 输入维度由 obs_dim 动态决定
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # 保持维度一致
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # 添加一个额外的层，继续保持维度一致
            nn.ReLU()
        )
        
        # 策略头 (输出动作的 logits)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim) # 输出维度由 action_dim 动态决定 (现在是 2)
        )
        # 值函数头 (输出状态的值函数估计)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1) # 值函数输出一个标量
        )
        
    def forward(self, x):
        shared_features = self.shared(x)
        policy_logits, value = self.policy_head(shared_features), self.value_head(shared_features)
        return policy_logits, value
    
    def get_action_and_value(self, x, action=None):
        policy_logits, value = self.forward(x)
        probs = Categorical(logits=policy_logits)
        if action is None:
            action = probs.sample()
        # 返回采样的动作、该动作的对数概率、策略的熵和状态价值
        return action, probs.log_prob(action), probs.entropy(), value