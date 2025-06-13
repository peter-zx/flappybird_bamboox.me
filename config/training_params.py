# === 文件：训练参数定义 ===
# 位置：E:\ai_work\Flappy Bird\config\training_params.py
# 用途：定义 PPO 算法的训练超参数和环境配置

class TrainingConfig:
    """训练配置类，用于定义 PPO 算法的超参数及环境设置"""
    
    def __init__(self):
        # --- 环境配置 ---
        self.env_name = "FlappyBird-v0" # 指定 Gymnasium 环境名称
        self.n_envs = 16                 # 并行环境数量，用于加速数据收集 (建议从 16 开始，根据硬件调整)
        self.render_mode = "rgb_array"   # 环境渲染模式 ("human" for display, "rgb_array" for no display)
                                         # 训练时通常使用 "rgb_array" 以提高速度

        # --- 算法选择 ---
        self.algorithm = "ppo"           # 默认算法：'ppo' 或 'dqn' (需在 train_flappy.py 中匹配)

        # --- 核心训练超参数 ---
        self.lr = 3e-4                   # 学习率，控制优化器更新模型参数的步长
        self.gamma = 0.99                # 折扣因子，衡量未来奖励的重要性 (0.99 常用)
        
        # --- PPO 特定超参数 ---
        self.gae_lambda = 0.95           # GAE（广义优势估计）的 λ 参数，用于 PPO 的优势计算 (0.95 常用)
        self.clip_coef = 0.2             # PPO 的剪切系数，限制策略更新的幅度 (0.2 常用)
        self.vf_coef = 0.5               # PPO 的值函数损失系数，平衡策略损失和值函数损失 (0.5 常用)
        self.ent_coef = 0.05             # PPO 的熵损失系数，鼓励探索 (0.01-0.05 常用，0.05 探索性更强)

        # --- 训练进度控制 ---
        self.total_timesteps = 100000    # 总训练步数 (建议 500万步或更高)
        self.start_timesteps = 0         # 起始步数，允许从某个时间步恢复训练 (从 0 开始表示全新训练)
        self.save_interval = 20000       # 模型保存间隔步数 (每 2万步保存一次)
        self.log_interval_episodes = 10  # 每隔多少个 episode 打印一次日志和平均奖励

        # ==================== 模型保存路径 (训练时只保存，不在这里加载) ====================
        self.model_save_base_dir = "trained_models"  
        
        # **重要：这里始终为 None**。因为恢复训练的路径会通过用户输入动态设置。
        self.load_model_path = None