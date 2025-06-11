# 数据收集

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

def collect_rollouts(env, network, device, n_envs, num_steps=2048):
    observations = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    
    obs = env.reset()[0] if hasattr(env, 'reset') else env.reset()
    obs = torch.FloatTensor(obs).to(device)
    
    episode_rewards = np.zeros(n_envs)
    episode_lengths = np.zeros(n_envs, dtype=int)
    
    for step in range(num_steps):
        with torch.no_grad():
            action, log_prob, _, value = network.get_action_and_value(obs)
            action = action.cpu().numpy() if n_envs == 1 else action.cpu().numpy()
        next_obs, reward, done, truncated, _ = env.step(action)
        next_obs = torch.FloatTensor(next_obs).to(device)
        
        observations.append(obs.cpu().numpy())
        actions.append(action)
        rewards.append(reward)
        dones.append(np.logical_or(done, truncated))
        values.append(value.cpu().numpy().squeeze())
        log_probs.append(log_prob.cpu().numpy())
        
        for i in range(n_envs):
            episode_rewards[i] += reward[i] if isinstance(reward, (list, np.ndarray)) else reward
            episode_lengths[i] += 1
            if dones[-1][i]:
                print(f"Env {i} Episode 完成! 奖励: {episode_rewards[i]:.2f}, 长度: {episode_lengths[i]}")
                episode_rewards[i] = 0
                episode_lengths[i] = 0
        
        obs = next_obs
    
    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'values': np.array(values),
        'log_probs': np.array(log_probs)
    }