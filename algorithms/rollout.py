# === 文件：数据收集 (Rollout) ===
# 位置：E:\ai_work\Flappy Bird\algorithms\rollout.py
# 用途：负责在环境中收集数据，形成一个批次供 PPO 算法训练

import torch
import numpy as np

def collect_rollouts(env, network, device, n_envs, num_steps_per_rollout=2048): # num_steps_per_rollout 可以调整
    """
    在环境中收集指定步数的数据。
    Args:
        env: Gymnasium 环境实例 (可以是向量化环境)
        network: PPO 神经网络
        device: PyTorch 设备 ('cpu' 或 'cuda')
        n_envs: 并行环境数量
        num_steps_per_rollout: 每个 rollout 收集的总步数 ( across all environments )
                                例如，如果 n_envs=16, num_steps_per_rollout=2048，则每个环境大约收集 128 步
    Returns:
        包含收集到的数据的字典
    """
    observations = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = [] # 记录每个时间步的 done 状态

    # 用于追踪每个环境的 episode 奖励和长度
    episode_rewards_in_rollout = []
    episode_lengths_in_rollout = []
    
    # 获取当前环境的观测值 (如果 env 是 SyncVectorEnv，则 obs 是一个批次)
    # env.reset() 返回 (obs, info)
    obs_np, info = env.reset() 
    # 如果是单个环境，obs_np 是 (obs_dim,)，需要扩展维度以匹配向量化环境的期望 (1, obs_dim)
    if n_envs == 1 and len(obs_np.shape) == 1:
        obs_np = np.expand_dims(obs_np, axis=0) # 使其成为 (1, obs_dim)

    # 初始化每个环境的当前奖励和长度计数器
    current_episode_rewards = np.zeros(n_envs)
    current_episode_lengths = np.zeros(n_envs)

    num_timesteps_collected = 0 # 追踪此 rollout 收集到的总步数

    # 在收集步数内循环
    for step in range(num_steps_per_rollout // n_envs): # 每个环境收集的步数
        # 将观测转换为 Tensor 并移到设备上
        obs_tensor = torch.FloatTensor(obs_np).to(device)

        with torch.no_grad():
            action_tensor, log_prob_tensor, _, value_tensor = network.get_action_and_value(obs_tensor)
        
        action_np = action_tensor.cpu().numpy()
        log_prob_np = log_prob_tensor.cpu().numpy()
        value_np = value_tensor.cpu().numpy().squeeze() # value_tensor 可能是 (n_envs, 1)

        # 在环境中执行动作
        next_obs_np, reward_np, done_np, truncated_np, info = env.step(action_np)

        # 记录数据
        observations.append(obs_np)
        actions.append(action_np)
        log_probs.append(log_prob_np)
        rewards.append(reward_np)
        values.append(value_np)
        
        # done 或 truncated 都表示回合结束
        # 对于 Gymnasium，done 表示环境达到终止状态，truncated 表示达到时间限制
        # 我们将两者都视为回合结束
        dones_combined = done_np | truncated_np
        dones.append(dones_combined)

        current_episode_rewards += reward_np
        current_episode_lengths += 1
        num_timesteps_collected += n_envs # 每次 step，所有环境都向前推进一步

        # 处理回合结束 (对于并行环境)
        for i, is_done in enumerate(dones_combined):
            if is_done:
                episode_rewards_in_rollout.append(current_episode_rewards[i])
                episode_lengths_in_rollout.append(current_episode_lengths[i])
                # 重置完成的环境
                reset_obs, reset_info = env.envs[i].reset() # 直接调用单个环境的 reset
                # 将重置后的观测值放回到正确的批次位置
                # 注意：如果 reset_obs 是 (obs_dim,)，需要 expand_dims
                if len(reset_obs.shape) == 1:
                    next_obs_np[i] = np.expand_dims(reset_obs, axis=0)
                else:
                    next_obs_np[i] = reset_obs
                
                # 重置对应环境的计数器
                current_episode_rewards[i] = 0
                current_episode_lengths[i] = 0
                
                # 打印单次 episode 完成信息 (仅供调试)
                # print(f"Env {i} Episode 完成! 奖励: {episode_rewards_in_rollout[-1]:.2f}, 长度: {episode_lengths_in_rollout[-1]}")
        
        # 更新 obs_np 到下一个状态
        obs_np = next_obs_np
    
    # 转换为 numpy 数组
    b_obs = np.array(observations) # (steps, n_envs, obs_dim)
    b_actions = np.array(actions) # (steps, n_envs)
    b_logprobs = np.array(log_probs) # (steps, n_envs)
    b_rewards = np.array(rewards) # (steps, n_envs)
    b_values = np.array(values) # (steps, n_envs)
    b_dones = np.array(dones) # (steps, n_envs)
    
    # 还需要最后一个 next_obs 来计算 GAE 的 next_value
    # next_obs_np 已经是最后一个时间步的观测 (next_obs)
    # 如果有环境在 rollout 结束时还没有结束，next_obs_np 就是它们的最新观测
    # 如果环境结束了，它已经被重置，next_obs_np 包含的是重置后的初始观测
    # 统一使用循环结束时的 obs_np 作为 next_observations
    b_next_observations = obs_np # (n_envs, obs_dim)

    # 返回所有收集到的数据
    return {
        'observations': b_obs,
        'actions': b_actions,
        'log_probs': b_logprobs,
        'rewards': b_rewards,
        'values': b_values,
        'dones': b_dones,
        'next_observations': b_next_observations, # 最后一个时间步的观测，用于GAE
        'episode_rewards_in_rollout': episode_rewards_in_rollout,
        'episode_lengths_in_rollout': episode_lengths_in_rollout,
        'num_timesteps_collected': num_timesteps_collected # <-- 添加这个
    }