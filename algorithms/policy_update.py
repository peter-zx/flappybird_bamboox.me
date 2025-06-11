import torch
import torch.nn.functional as F

def update_policy(network, optimizer, batch_data, device, clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, num_epochs=4, batch_size=64):
    # 展平多环境数据
    observations = torch.FloatTensor(batch_data['observations']).to(device)  # (num_steps, n_envs, obs_dim)
    actions = torch.LongTensor(batch_data['actions']).to(device)  # (num_steps, n_envs)
    old_log_probs = torch.FloatTensor(batch_data['log_probs']).to(device)  # (num_steps * n_envs,)
    advantages = torch.FloatTensor(batch_data['advantages']).to(device)  # (num_steps, n_envs)
    returns = torch.FloatTensor(batch_data['returns']).to(device)  # (num_steps, n_envs)
    
    # 展平为 (num_steps * n_envs,) 以匹配批量处理
    total_steps = observations.size(0)
    n_envs = observations.size(1)
    observations = observations.view(total_steps * n_envs, -1)
    actions = actions.view(total_steps * n_envs)
    old_log_probs = old_log_probs.view(-1)
    advantages = advantages.view(-1)
    returns = returns.view(-1)
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for epoch in range(num_epochs):
        indices = torch.randperm(len(observations))
        
        for start in range(0, len(observations), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batch_obs = observations[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            
            # 获取新的策略输出，确保维度匹配
            _, new_log_probs, entropy, new_values = network.get_action_and_value(batch_obs, batch_actions)
            new_log_probs = new_log_probs.view(-1)  # 展平为 (batch_size,)
            
            # 检查维度
            if new_log_probs.shape != batch_old_log_probs.shape:
                raise ValueError(f"new_log_probs shape {new_log_probs.shape} does not match batch_old_log_probs shape {batch_old_log_probs.shape}")
            
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
            entropy_loss = -entropy.mean()
            
            total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()