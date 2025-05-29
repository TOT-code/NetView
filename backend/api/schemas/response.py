"""
API响应数据模型
定义所有API端点的响应数据结构
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelAnalysisResponse(BaseModel):
    """模型分析响应"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    message: str = Field(..., description="状态消息")
    created_at: datetime = Field(..., description="创建时间")
    estimated_duration: Optional[int] = Field(None, description="预估完成时间(秒)")
    
    # 如果分析完成，包含模型信息
    model_info: Optional[Dict[str, Any]] = Field(None, description="模型基本信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_12345678-1234-1234-1234-123456789abc",
                "status": "processing",
                "message": "正在分析模型结构...",
                "created_at": "2025-05-28T11:22:00Z",
                "estimated_duration": 30,
                "model_info": {
                    "name": "SimpleCNN",
                    "graph_type": "resnet",
                    "total_parameters": 1276234
                }
            }
        }

class VisualizationResponse(BaseModel):
    """可视化数据响应"""
    task_id: str = Field(..., description="任务ID")
    model_name: str = Field(..., description="模型名称")
    graph_type: str = Field(..., description="图类型")
    
    # 基础图结构
    nodes: List[Dict[str, Any]] = Field(..., description="节点列表")
    edges: List[Dict[str, Any]] = Field(..., description="边列表")
    
    # 元数据
    metadata: Dict[str, Any] = Field(..., description="图元数据")
    
    # 增强分析数据
    dynamic_info: Optional[Dict[str, Any]] = Field(None, description="动态分析信息")
    architecture_patterns: Optional[Dict[str, Any]] = Field(None, description="架构模式信息")
    tensor_flow: Optional[Dict[str, Any]] = Field(None, description="Tensor流信息")
    
    # 统计信息
    statistics: Dict[str, Any] = Field(..., description="统计信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_12345678-1234-1234-1234-123456789abc",
                "model_name": "SimpleCNN",
                "graph_type": "resnet",
                "nodes": [
                    {
                        "id": "conv1",
                        "type": "Conv2d",
                        "label": "Conv2d(3, 32, kernel_size=(3, 3))",
                        "parameters": {"in_channels": 3, "out_channels": 32}
                    }
                ],
                "edges": [
                    {
                        "source": "conv1",
                        "target": "relu1",
                        "connection_type": "sequential"
                    }
                ],
                "metadata": {
                    "total_parameters": 1276234,
                    "model_size_mb": 4.87
                },
                "statistics": {
                    "node_count": 9,
                    "edge_count": 11,
                    "layer_types": {"Conv2d": 3, "ReLU": 3, "Linear": 3}
                }
            }
        }

class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    progress: Optional[float] = Field(None, description="进度百分比 (0-100)", ge=0, le=100)
    message: str = Field(..., description="状态消息")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    
    # 错误信息
    error: Optional[str] = Field(None, description="错误信息")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    
    # 结果链接
    result_url: Optional[str] = Field(None, description="结果获取URL")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_12345678-1234-1234-1234-123456789abc",
                "status": "completed",
                "progress": 100.0,
                "message": "模型分析完成",
                "created_at": "2025-05-28T11:22:00Z",
                "updated_at": "2025-05-28T11:22:30Z",
                "completed_at": "2025-05-28T11:22:30Z",
                "result_url": "/api/v1/models/task_12345678-1234-1234-1234-123456789abc/visualization"
            }
        }

class ExportResponse(BaseModel):
    """导出响应"""
    task_id: str = Field(..., description="任务ID")
    export_format: str = Field(..., description="导出格式")
    file_url: Optional[str] = Field(None, description="文件下载URL")
    file_data: Optional[str] = Field(None, description="文件数据（Base64编码）")
    file_size: Optional[int] = Field(None, description="文件大小（字节）")
    expires_at: Optional[datetime] = Field(None, description="文件过期时间")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_12345678-1234-1234-1234-123456789abc",
                "export_format": "png",
                "file_url": "/api/v1/exports/download/export_abc123.png",
                "file_size": 102400,
                "expires_at": "2025-05-29T11:22:00Z"
            }
        }

class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    timestamp: datetime = Field(..., description="错误时间")
    request_id: Optional[str] = Field(None, description="请求ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "输入数据验证失败",
                "details": {
                    "field": "code",
                    "issue": "代码必须包含PyTorch模型定义"
                },
                "timestamp": "2025-05-28T11:22:00Z",
                "request_id": "req_12345678"
            }
        }

class ConfigResponse(BaseModel):
    """配置响应"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作消息")
    config: Dict[str, Any] = Field(..., description="当前配置")
    
class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    timestamp: datetime = Field(..., description="检查时间")
    version: str = Field(..., description="API版本")
    uptime: Optional[int] = Field(None, description="运行时间（秒）")
    system_info: Optional[Dict[str, Any]] = Field(None, description="系统信息")

class BatchAnalysisResponse(BaseModel):
    """批量分析响应"""
    batch_id: str = Field(..., description="批次ID")
    total_models: int = Field(..., description="模型总数")
    task_ids: List[str] = Field(..., description="任务ID列表")
    estimated_duration: Optional[int] = Field(None, description="预估总时间")
    created_at: datetime = Field(..., description="创建时间")

class ModelComparisonResponse(BaseModel):
    """模型对比响应"""
    comparison_id: str = Field(..., description="对比ID")
    model_a_task_id: str = Field(..., description="模型A任务ID")
    model_b_task_id: str = Field(..., description="模型B任务ID")
    comparison_data: Optional[Dict[str, Any]] = Field(None, description="对比结果")
    status: TaskStatus = Field(..., description="对比状态")
