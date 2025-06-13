# === 文件：策略更新模块 ===
# 位置：E:\ai_work\Flappy Bird\algorithms\policy_update.py

import torch
import torch.nn as nn
import torch.optim as optim

# 导入必要的模块 (确保这些导入在你项目中是有效的)
from algorithms.networks import PPONetwork 
from algorithms.rollout import RolloutStorage 

def ppo_update(
    actor_critic: nn.Module,
    optimizer: optim.Adam,
    rollouts: RolloutStorage,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    num_mini_batch: int,
    update_epochs: int,
    device: torch.device
):
    """
    PPO 算法的核心更新逻辑。
    """
    # 计算优势 (Adjusted for GAE)
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    # 优势归一化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 打平 rollouts 数据，准备进行 mini-batch 更新
    # 将数据从 (num_steps, num_envs, ...) 变为 (num_steps * num_envs, ...)
    flat_obs = rollouts.observations[:-1].view(-1, rollouts.observations.shape[-1])
    flat_actions = rollouts.actions.view(-1, 1)
    flat_returns = rollouts.returns[:-1].view(-1, 1) # 从 returns[:-1] 获取目标值
    flat_old_value_preds = rollouts.value_preds[:-1].view(-1, 1)
    flat_old_log_probs = rollouts.log_probs.view(-1, 1)
    flat_advantages = advantages.view(-1, 1)

    # 组合所有数据并打乱，用于创建 mini-batch
    # 使用 LongTensor for actions, as_tensor for others
    data = torch.cat((flat_obs, 
                      flat_actions.long(), 
                      flat_returns, 
                      flat_old_value_preds, 
                      flat_old_log_probs, 
                      flat_advantages), dim=1)

    batch_size = data.shape[0] // num_mini_batch
    
    for _ in range(update_epochs):
        # 随机打乱数据
        perm = torch.randperm(data.shape[0])
        shuffled_data = data[perm]

        for i in range(num_mini_batch):
            start = i * batch_size
            end = (i + 1) * batch_size
            mini_batch = shuffled_data[start:end]

            # 分离 mini-batch 数据
            mini_obs = mini_batch[:, 0:rollouts.observations.shape[-1]]
            mini_actions = mini_batch[:, rollouts.observations.shape[-1]].long().unsqueeze(1)
            mini_returns = mini_batch[:, rollouts.observations.shape[-1] + 1].unsqueeze(1)
            mini_old_value_preds = mini_batch[:, rollouts.observations.shape[-1] + 2].unsqueeze(1)
            mini_old_log_probs = mini_batch[:, rollouts.observations.shape[-1] + 3].unsqueeze(1)
            mini_advantages = mini_batch[:, rollouts.observations.shape[-1] + 4].unsqueeze(1)
            
            # 计算当前策略的输出
            new_value, new_dist = actor_critic(mini_obs)
            new_log_probs = new_dist.log_prob(mini_actions.squeeze(1)).unsqueeze(1)
            
            # 计算 PPO 损失
            ratio = torch.exp(new_log_probs - mini_old_log_probs)
            surr1 = ratio * mini_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mini_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 值函数损失
            value_loss = (new_value - mini_returns).pow(2).mean() # MSE 损失

            # 熵损失 (鼓励探索)
            entropy_loss = new_dist.entropy().mean()

            # 总损失
            total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪 (防止梯度爆炸)
            nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5) 
            optimizer.step()