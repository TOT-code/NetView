"""
NetView API模块
提供RESTful API服务，支持模型解析、数据传输和状态管理
"""

__version__ = "0.2.0"
__author__ = "NetView Team"

from .main import app

__all__ = ["app"]
