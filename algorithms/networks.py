# 网络定义

import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, action_dim)
        )
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