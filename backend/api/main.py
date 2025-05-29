"""
FastAPI主应用
NetView RESTful API服务入口
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.api.routers import models, visualization, config, export, tasks
from backend.api.middleware.error_handler import add_error_handlers

# 创建FastAPI应用实例
app = FastAPI(
    title="NetView API",
    description="AI驱动的PyTorch模型深度分析API服务",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],  # 明确指定允许的源
    allow_credentials=False,  # 设置为False以避免冲突
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 添加错误处理
add_error_handlers(app)

# 注册路由
app.include_router(models.router, prefix="/api/v1", tags=["模型解析"])
app.include_router(visualization.router, prefix="/api/v1", tags=["可视化"])
app.include_router(config.router, prefix="/api/v1", tags=["配置管理"])
app.include_router(export.router, prefix="/api/v1", tags=["导出功能"])
app.include_router(tasks.router, prefix="/api/v1", tags=["任务管理"])

@app.get("/", tags=["根路径"])
async def root():
    """API根路径，返回服务信息"""
    return {
        "name": "NetView API",
        "version": "0.2.0",
        "description": "AI驱动的PyTorch模型深度分析API服务",
        "docs": "/api/docs",
        "features": [
            "模型结构解析",
            "动态图分析", 
            "架构模式识别",
            "Tensor流追踪",
            "可视化数据生成",
            "多格式导出"
        ]
    }

@app.get("/api/health", tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": "2025-05-28T11:21:00Z",
        "version": "0.2.0"
    }

@app.get("/api/v1/features", tags=["功能列表"])
async def get_features():
    """获取API支持的功能列表"""
    return {
        "analysis_features": {
            "ast_analysis": "AST静态分析",
            "pytorch_inspection": "PyTorch运行时内省",
            "dynamic_analysis": "动态图分析",
            "architecture_patterns": "架构模式识别",
            "tensor_flow_analysis": "Tensor流分析"
        },
        "supported_models": [
            "CNN (卷积神经网络)",
            "ResNet (残差网络)",
            "DenseNet (密集连接网络)",
            "自定义PyTorch模型"
        ],
        "export_formats": ["JSON", "PNG", "SVG"],
        "analysis_options": {
            "enable_dynamic": "启用动态分析",
            "enable_patterns": "启用架构模式识别",
            "enable_tensor_flow": "启用Tensor流分析",
            "custom_input_shape": "自定义输入形状"
        }
    }

# 添加一个简单的测试端点
@app.get("/api/v1/test", tags=["测试"])
async def test_cors():
    """测试CORS配置"""
    return {"message": "CORS测试成功", "status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
