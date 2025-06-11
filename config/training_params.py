# === 文件：训练参数定义 ===
# 位置：E:\ai_work\Flappy Bird\config\training_params.py
# 用途：定义训练所需的超参数和环境配置，供 PPO 和 DQN 等算法使用

class TrainingConfig:
    """训练配置类，用于定义 PPO 和 DQN 算法的超参数及环境设置"""
    
    def __init__(self):
        # 环境名称，指定 Gymnasium 环境
        # 默认值：'FlappyBird-v0'，指向你的 flappy-bird-gymnasium 环境
        self.env_name = "FlappyBird-v0"
        
        # 算法选择，决定使用哪种强化学习算法
        # 默认值：'ppo'，支持 'ppo'（近端策略优化）或 'dqn'（深度 Q 学习）
        self.algorithm = "ppo"  # 默认算法
        
        # 学习率，控制优化器更新模型参数的步长
        # 默认值：3e-4（0.0003），适用于 PPO 和 DQN 的 Adam 优化器
        self.lr = 3e-4
        
        # 折扣因子，衡量未来奖励的重要性
        # 默认值：0.99，表示长期奖励对当前决策有较高影响
        self.gamma = 0.99
        
        # GAE（广义优势估计）的 λ 参数，用于 PPO 的优势计算
        # 默认值：0.95，平衡偏差和方差，仅 PPO 使用
        self.gae_lambda = 0.95
        
        # PPO 的剪切系数，限制策略更新的幅度
        # 默认值：0.2，确保 PPO 训练稳定，仅 PPO 使用
        self.clip_coef = 0.2
        
        # PPO 的值函数损失系数，平衡策略损失和值函数损失
        # 默认值：0.5，仅 PPO 使用
        self.vf_coef = 0.5
        
        # PPO 的熵损失系数，鼓励探索
        # 默认值：0.01，仅 PPO 使用
        self.ent_coef = 0.05
        
        # 并行环境数量，用于加速数据收集
        # 默认值：4，PPO 使用多环境，DQN 通常为 1
        self.n_envs = 8
        
        # 总训练步数，指定训练的总时间步
        # 默认值：1000000（100万步），适用于 PPO 和 DQN
        self.total_timesteps = 500000
        
        # 起始步数，允许从某个时间步恢复训练
        # 默认值：0，表示从头开始训练
        self.start_timesteps = 1000000