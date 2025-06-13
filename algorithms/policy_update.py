# === 文件：策略更新 ===
# 位置：E:\ai_work\Flappy Bird\algorithms\policy_update.py
# 用途：定义 PPO 算法中策略和值函数的更新逻辑

import torch
import torch.nn.functional as F
import numpy as np

def update_policy(network, optimizer, batch_data, device, clip_coef, vf_coef, ent_coef, num_epochs=10, mini_batch_size=256):
    """
    更新 PPO 网络的策略和值函数。
    Args:
        network: PPO 神经网络
        optimizer: 优化器
        batch_data: 包含收集到的数据的字典 (obs, actions, log_probs, rewards, values, dones, advantages, returns)
        device: PyTorch 设备 ('cpu' 或 'cuda')
        clip_coef: PPO 裁剪系数
        vf_coef: 值函数损失系数
        ent_coef: 熵损失系数
        num_epochs: PPO 更新的 epoch 数
        mini_batch_size: 每个 mini-batch 的大小
    Returns:
        包含平均损失的字典
    """
    
    # 从 batch_data 中提取数据并转换为 Tensor
    b_obs = torch.FloatTensor(batch_data['observations']).to(device).reshape(-1, network.obs_dim)
    b_actions = torch.LongTensor(batch_data['actions']).to(device).reshape(-1) # actions 应该是 LongTensor
    b_logprobs = torch.FloatTensor(batch_data['log_probs']).to(device).reshape(-1)
    b_advantages = torch.FloatTensor(batch_data['advantages']).to(device).reshape(-1)
    b_returns = torch.FloatTensor(batch_data['returns']).to(device).reshape(-1)
    b_values = torch.FloatTensor(batch_data['values']).to(device).reshape(-1) # 原始值函数估计，用于值函数裁剪

    # 优势归一化 (可选但推荐)
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    # 扁平化数据以进行 batch_size 划分
    # total_samples = b_obs.shape[0] # obs 已经被 reshape 成 (total_samples, obs_dim)
    
    # 记录每个 epoch 的平均损失
    avg_policy_loss = 0
    avg_value_loss = 0
    avg_entropy_loss = 0
    
    # PPO 训练循环
    for epoch in range(num_epochs):
        # 随机打乱索引并生成 mini-batches
        # 确保 mini_batch_size 不大于总样本数
        actual_mini_batch_size = min(mini_batch_size, b_obs.shape[0])
        
        # 获得所有样本的随机索引
        indices = torch.randperm(b_obs.shape[0], device=device)
        
        # 遍历 mini-batches
        for start_idx in range(0, b_obs.shape[0], actual_mini_batch_size):
            end_idx = start_idx + actual_mini_batch_size
            batch_indices = indices[start_idx:end_idx]

            mb_obs = b_obs[batch_indices]
            mb_actions = b_actions[batch_indices]
            mb_old_logprobs = b_logprobs[batch_indices]
            mb_advantages = b_advantages[batch_indices]
            mb_returns = b_returns[batch_indices]
            mb_old_values = b_values[batch_indices] # 原始值函数估计

            # 计算当前策略和值
            new_policy_logits, new_values = network(mb_obs)
            dist = torch.distributions.Categorical(logits=new_policy_logits)
            new_logprobs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean() # 熵

            # 策略损失
            ratio = torch.exp(new_logprobs - mb_old_logprobs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 值函数损失 (使用裁剪和MSE)
            new_values_clipped = mb_old_values + torch.clamp(
                new_values.squeeze() - mb_old_values, 
                -clip_coef, 
                clip_coef
            )
            value_loss_clipped = F.mse_loss(new_values_clipped, mb_returns)
            value_loss_unclipped = F.mse_loss(new_values.squeeze(), mb_returns)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean() # 0.5 是惯例系数

            # 总损失
            total_loss = policy_loss + value_loss * vf_coef - entropy * ent_coef

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5) # 梯度裁剪 (可选，但推荐)
            optimizer.step()

            # 累积损失
            avg_policy_loss += policy_loss.item()
            avg_value_loss += value_loss.item()
            avg_entropy_loss += entropy.item()

    # 计算平均损失
    num_minibatches = (b_obs.shape[0] + actual_mini_batch_size - 1) // actual_mini_batch_size
    avg_policy_loss /= (num_epochs * num_minibatches)
    avg_value_loss /= (num_epochs * num_minibatches)
    avg_entropy_loss /= (num_epochs * num_minibatches)

    # 返回损失信息
    return {
        'policy_loss': avg_policy_loss,
        'value_loss': avg_value_loss,
        'entropy_loss': avg_entropy_loss
    }