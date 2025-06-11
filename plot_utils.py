# 新增：绘图工具，放置在根目录

# === 新增文件：绘图工具 ===
# 位置：E:\ai_work\Flappy Bird\plot_utils.py
# 原因：实现训练/测试数据可视化，解决“没有训练数据可以绘制”问题
import matplotlib.pyplot as plt
import os

def plot_reward_bar(episodes, rewards, save_path):
    """绘制奖励分布柱状图"""
    plt.figure(figsize=(10, 6))
    plt.bar(episodes, rewards, color="skyblue")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode")
    plt.grid(True, axis="y")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_training_progress(rewards, lengths, save_path):
    """绘制训练进度（奖励和步数），与 training_progress.png 一致"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(lengths)
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()