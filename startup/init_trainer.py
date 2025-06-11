# === Modified File: Initialize Trainer ===
# Purpose: Support PPO and DQN switching, retain environment registration and CUDA check
import gymnasium as gym
from gymnasium.envs.registration import register
from algorithms.trainer import PPOTrainer
from algorithms.dqn import DQNTrainer
from config.training_params import TrainingConfig
import torch

# === Retained: Manual environment registration ===
# Reason: Ensure FlappyBird-v0 is registered
register(
    id='FlappyBird-v0',
    entry_point='flappy_bird_gymnasium.envs:FlappyBirdEnv',
)

def initialize_trainer(config: TrainingConfig):
    # === Retained: CUDA availability check ===
    # Reason: Provide hardware information
    if torch.cuda.is_available():
        print(f"CUDA 可用! 设备: {torch.cuda.get_device_name()}")
        print(f"CUDA 版本: {torch.version.cuda}")
    else:
        print("CUDA 不可用，使用 CPU 训练")
    
    # === Modified: Support PPO and DQN switching ===
    # Reason: Allow algorithm selection via config.algorithm
    if config.algorithm == "ppo":
        trainer = PPOTrainer(
            env_name=config.env_name,
            lr=config.lr,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_coef=config.clip_coef,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            n_envs=config.n_envs
        )
    elif config.algorithm == "dqn":
        trainer = DQNTrainer(
            env_name=config.env_name,
            lr=config.lr,
            gamma=config.gamma,
            n_envs=config.n_envs
        )
    else:
        raise ValueError(f"不支持的算法: {config.algorithm}")
    
    return trainer
# === End Modification ===