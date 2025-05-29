"""
CORS中间件配置
处理跨域资源共享
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

def setup_cors(app: FastAPI, 
               allowed_origins: List[str] = None,
               allowed_methods: List[str] = None,
               allowed_headers: List[str] = None,
               allow_credentials: bool = True):
    """设置CORS中间件"""
    
    # 默认配置
    if allowed_origins is None:
        allowed_origins = [
            "http://localhost:3000",  # React开发服务器
            "http://localhost:8080",  # 前端开发服务器
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
            "http://localhost",
            "http://127.0.0.1",
        ]
    
    if allowed_methods is None:
        allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    
    if allowed_headers is None:
        allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID",
            "X-API-Key",
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
        expose_headers=["X-Request-ID", "X-Process-Time"],
        max_age=3600,  # 预检请求缓存时间
    )

def setup_production_cors(app: FastAPI, production_origins: List[str]):
    """设置生产环境CORS"""
    
    # 生产环境更严格的CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=production_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=[
            "Accept",
            "Content-Type", 
            "Authorization",
            "X-Request-ID"
        ],
        expose_headers=["X-Request-ID", "X-Process-Time"],
        max_age=3600,
    )

def setup_development_cors(app: FastAPI):
    """设置开发环境CORS（允许所有来源）"""
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 开发环境允许所有来源
        allow_credentials=False,  # 允许所有来源时必须设为False
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )
