# === 文件：绘图工具函数 ===
# 位置：E:\ai_work\Flappy Bird\plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_progress(rewards, lengths, save_path):
    """
    绘制训练奖励和回合长度的曲线图。
    rewards: 列表，每个元素是每个 episode 的总奖励。
    lengths: 列表，每个元素是每个 episode 的总长度。
    save_path: 字符串，保存图表的完整文件路径（包括文件名和扩展名）。
    """
    if not rewards or not lengths:
        print("没有训练数据可供绘图。")
        return

    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    episodes = np.arange(len(rewards)) + 1

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制奖励曲线
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(episodes, rewards, color=color, alpha=0.7, label='Total Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个 Y 轴绘制回合长度
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Episode Length', color=color)  
    ax2.plot(episodes, lengths, color=color, alpha=0.7, linestyle='--', label='Episode Length')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题和图例
    fig.tight_layout()  # 调整布局以避免重叠
    plt.title('Training Progress: Reward and Episode Length over Episodes')
    
    # 手动创建图例，因为有两个Y轴
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.grid(True)
    plt.savefig(save_path)
    plt.close(fig) # 关闭图形，防止内存泄漏
    # print(f"训练进度图已保存到: {save_path}") # 在 ppo.py 中打印，这里不再打印

def plot_test_reward_bar(test_rewards, save_path):
    """
    绘制测试奖励的柱状图。
    test_rewards: 列表，每个元素是每个测试 episode 的总奖励。
    save_path: 字符串，保存图表的完整文件路径。
    """
    if not test_rewards:
        print("没有测试数据可供绘图。")
        return
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    episodes = np.arange(len(test_rewards)) + 1
    
    plt.figure(figsize=(10, 6))
    plt.bar(episodes, test_rewards, color='skyblue')
    plt.xlabel('Test Episode')
    plt.ylabel('Total Reward')
    plt.title('Test Rewards per Episode')
    plt.xticks(episodes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()