# === 文件：测试参数定义 ===
# 位置：E:\ai_work\Flappy Bird\config\test_params.py
# 用途：定义测试所需的参数

class TestConfig:
    """测试配置类，用于定义测试过程的参数"""
    
    def __init__(self):
        # --- 环境配置 (测试环境) ---
        self.env_name = "FlappyBird-v0" # 指定 Gymnasium 环境名称 (与训练环境保持一致)
        self.n_envs = 1 # 测试时通常只需要 1 个环境 (因为要 human 渲染)

        # ==================== 模型加载路径 (测试时加载) ====================
        # **重要：这里可以设置为 None，让 main_flappy.py 自动查找最新模型。
        #        或者设置为一个默认的特定模型路径，如果你想总是测试它。**
        self.load_model_path = None 
        
        # 当 self.load_model_path 为 None 时，自动查找模型的根目录
        self.model_search_base_dir = "trained_models" 
        
        # --- 测试行为 ---
        self.test_episodes_after_train = 5 # 测试进行的回合数
        self.test_render_mode = "human"    # 测试时的渲染模式 ("human" 显示画面, "rgb_array" 不显示)