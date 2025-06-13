# === 测试脚本 ===
# 位置：E:\ai_work\Flappy Bird\test_flappy.py
import os
import json
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
import pygame
from startup.init_trainer import initialize_trainer
from config.training_params import TrainingConfig
from plot_utils import plot_reward_bar

PROJECT_ROOT = r"E:\ai_work\Flappy Bird"
MODEL_DIR = os.path.join(PROJECT_ROOT, "flappy_bird_models_20250610_184044")
MODEL_PATH = os.path.join(MODEL_DIR, "model_step_1007616.pth")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TEST_LOG_PATH = os.path.join(LOG_DIR, "test_log.json")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def render_multi_envs(frames, screen, env_width, env_height, num_envs):
    """在 2x2 网格内渲染 4 个环境画面"""
    for i, frame in enumerate(frames):
        frame = np.transpose(frame, (1, 0, 2))  # (H, W, C) -> (W, H, C)
        surface = pygame.surfarray.make_surface(frame)
        surface = pygame.transform.scale(surface, (env_width // 2, env_height // 2))
        x = (i % 2) * (env_width // 2)
        y = (i // 2) * (env_height // 2)
        screen.blit(surface, (x, y))
    pygame.display.flip()

def test_model(trainer, env, num_episodes=10, num_envs=4, device="cuda"):
    pygame.init()
    env_width, env_height = 288, 512
    screen = pygame.display.set_mode((env_width, env_height))
    pygame.display.set_caption("Flappy Bird Multi-Agent Test")

    clock = pygame.time.Clock()

    test_results = []
    episode_counts = np.zeros(num_envs, dtype=int)
    total_rewards = np.zeros(num_envs)
    total_steps = np.zeros(num_envs, dtype=int)
    done_episodes = 0
    
    obs = env.reset()[0]
    print(f"初始观测：{obs[0]}, 形状：{obs.shape}")
    obs = np.array(obs, dtype=np.float32)
    obs = obs / np.max(obs, axis=1, keepdims=True).clip(min=1e-6)

    while done_episodes < num_episodes * num_envs:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return test_results

        obs_tensor = torch.FloatTensor(obs).to(device)
        actions = []
        for i in range(num_envs):
            if episode_counts[i] < num_episodes:
                with torch.no_grad():
                    action_logits, _ = trainer.network(obs_tensor[i:i+1])
                    action_probs = torch.softmax(action_logits, dim=-1)
                    action = torch.multinomial(action_probs, 1).item() % 2
                actions.append(action)
            else:
                actions.append(0)
        actions = np.array(actions)

        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        dones = np.logical_or(terminated, truncated)
        next_obs = np.array(next_obs, dtype=np.float32)
        next_obs = next_obs / np.max(next_obs, axis=1, keepdims=True).clip(min=1e-6)

        for i in range(num_envs):
            if episode_counts[i] < num_episodes:
                total_rewards[i] += rewards[i]
                total_steps[i] += 1
                if dones[i]:
                    print(f"环境 {i+1} 回合 {episode_counts[i]+1}: 总奖励 = {total_rewards[i]:.2f}, 步数 = {total_steps[i]}")
                    test_results.append({
                        "env": int(i + 1),
                        "episode": int(episode_counts[i] + 1),
                        "reward": float(total_rewards[i]),
                        "steps": int(total_steps[i])
                    })
                    episode_counts[i] += 1
                    total_rewards[i] = 0
                    total_steps[i] = 0
                    done_episodes += 1
                    obs[i] = env.envs[i].reset()[0]
                    obs[i] = np.array(obs[i], dtype=np.float32)
                    obs[i] = obs[i] / obs[i].max() if obs[i].max() != 0 else obs[i]
                else:
                    obs[i] = next_obs[i]

        frames = [env.envs[i].render() for i in range(num_envs)]  # 移除 show_rays
        if episode_counts[0] == 0:
            print(f"渲染帧形状：{frames[0].shape}")
        render_multi_envs(frames, screen, env_width, env_height, num_envs)
        clock.tick(60)

    pygame.quit()
    with open(TEST_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4, ensure_ascii=False)
    
    return test_results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    num_envs = 4
    try:
        env = SyncVectorEnv([
            lambda: gym.make("FlappyBird-v0", render_mode="rgb_array")
            for _ in range(num_envs)
        ])
        print(f"动作空间：{env.envs[0].action_space}")
    except gym.error.NameNotFound:
        print("环境 FlappyBird-v0 未注册！")
        return
    
    config = TrainingConfig()
    config.algorithm = "ppo"
    trainer = initialize_trainer(config)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        try:
            trainer.network.load_state_dict(checkpoint["network_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"从 {MODEL_PATH} 加载模型，步数: {checkpoint['timesteps']}")
        except RuntimeError as e:
            print(f"加载模型失败：{e}")
            return
    else:
        print(f"模型文件 {MODEL_PATH} 不存在！")
        return
    
    test_results = test_model(trainer, env, num_episodes=10, num_envs=num_envs, device=device)
    
    rewards = [result["reward"] for result in test_results]
    episodes = [f"Env{result['env']}-Ep{result['episode']}" for result in test_results]
    plot_reward_bar(range(len(rewards)), rewards, save_path=os.path.join(PLOT_DIR, "test_reward_bar.png"))
    print(f"奖励分布图已保存至: {PLOT_DIR}\\test_reward_bar.png")
    
    avg_reward = np.mean(rewards)
    avg_steps = np.mean([result["steps"] for result in test_results])
    print(f"\n平均奖励: {avg_reward:.2f}, 平均步数: {avg_steps:.2f}")

if __name__ == "__main__":
    main()