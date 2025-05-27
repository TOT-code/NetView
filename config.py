"""
NetView 项目配置文件
包含所有模块的配置参数
"""

import os

# 基础配置
class Config:
    """基础配置类"""
    
    # 项目信息
    PROJECT_NAME = "NetView"
    VERSION = "0.1.0"
    DESCRIPTION = "PyTorch网络结构可视化工具"
    
    # 服务器配置
    HOST = os.environ.get('NETVIEW_HOST', 'localhost')
    PORT = int(os.environ.get('NETVIEW_PORT', 5000))
    DEBUG = os.environ.get('NETVIEW_DEBUG', 'True').lower() == 'true'
    
    # 安全配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'uploads'
    
    # 缓存配置
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/netview.log'

# 图形可视化配置
class VisualizationConfig:
    """可视化相关配置"""
    
    # 节点样式配置
    NODE_SIZE = {
        'min': 30,
        'max': 100,
        'default': 60
    }
    
    # 边样式配置
    EDGE_WIDTH = {
        'min': 1,
        'max': 5,
        'default': 2
    }
    
    # 层类型颜色映射
    LAYER_COLORS = {
        'Conv2d': '#FF6B6B',        # 卷积层 - 红色
        'Linear': '#4ECDC4',        # 全连接层 - 青色
        'MaxPool2d': '#45B7D1',     # 池化层 - 蓝色
        'AvgPool2d': '#45B7D1',     # 平均池化 - 蓝色
        'BatchNorm2d': '#96CEB4',   # 批归一化 - 绿色
        'BatchNorm1d': '#96CEB4',   # 批归一化 - 绿色
        'ReLU': '#FFEAA7',          # 激活函数 - 黄色
        'Sigmoid': '#FFEAA7',       # 激活函数 - 黄色
        'Tanh': '#FFEAA7',          # 激活函数 - 黄色
        'Dropout': '#DDA0DD',       # Dropout - 紫色
        'default': '#95A5A6'        # 默认 - 灰色
    }
    
    # 布局配置
    LAYOUT_OPTIONS = {
        'hierarchical': {
            'direction': 'TB',      # Top-Bottom
            'spacing': 100
        },
        'force_directed': {
            'iterations': 300,
            'spring_strength': 0.1
        }
    }

# 模型解析配置
class ParserConfig:
    """模型解析相关配置"""
    
    # 解析限制
    MAX_RECURSION_DEPTH = 10
    MAX_NODES = 1000
    TIMEOUT_SECONDS = 30
    
    # 忽略的模块
    IGNORE_MODULES = [
        'torch.nn.functional',
        '__builtin__',
        'builtins'
    ]
    
    # 自定义层映射
    CUSTOM_LAYER_MAPPING = {
        # 可以添加自定义层的映射关系
    }
    
    # 支持的PyTorch层类型
    SUPPORTED_LAYERS = [
        'Conv1d', 'Conv2d', 'Conv3d',
        'Linear',
        'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
        'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
        'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
        'ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU',
        'Dropout', 'Dropout2d', 'Dropout3d'
    ]

# 导出配置
class ExportConfig:
    """图像导出相关配置"""
    
    # 支持的导出格式
    SUPPORTED_FORMATS = ['png', 'svg', 'pdf']
    
    # 默认导出参数
    DEFAULT_WIDTH = 1200
    DEFAULT_HEIGHT = 800
    DEFAULT_DPI = 300
    
    # 文件大小限制
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# 根据环境选择配置
class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# 获取当前配置
def get_config():
    """获取当前环境的配置"""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])
