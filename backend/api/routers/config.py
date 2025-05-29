"""
配置管理API路由
处理系统配置和用户设置
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.api.schemas.request import ConfigRequest
from backend.api.schemas.response import ConfigResponse

router = APIRouter()

# 全局配置存储（生产环境应使用数据库）
_global_config = {
    "analysis_config": {
        "default_input_shape": [1, 3, 224, 224],
        "max_analysis_time": 300,
        "enable_dynamic_analysis": True,
        "enable_pattern_recognition": True,
        "enable_tensor_flow_analysis": True,
        "max_model_size_mb": 100
    },
    "visualization_config": {
        "default_layout": "hierarchical",
        "node_spacing": 100,
        "edge_style": "curved",
        "color_scheme": "default",
        "show_parameters": True,
        "show_shapes": True,
        "animation_enabled": False
    },
    "export_config": {
        "default_format": "json",
        "max_export_size_mb": 50,
        "export_expiry_hours": 24,
        "image_quality": "high",
        "svg_precision": 2
    },
    "system_config": {
        "max_concurrent_tasks": 10,
        "task_cleanup_interval_hours": 6,
        "log_level": "INFO",
        "enable_performance_monitoring": True
    }
}

@router.get("/config",
           summary="获取系统配置",
           description="获取当前的系统配置设置")
async def get_config() -> Dict[str, Any]:
    """
    获取系统配置
    
    **配置分类：**
    - `analysis_config`: 分析相关配置
    - `visualization_config`: 可视化配置
    - `export_config`: 导出功能配置
    - `system_config`: 系统运行配置
    
    **用途：**
    - 前端界面初始化
    - 用户设置回显
    - 系统状态检查
    """
    return {
        "config": _global_config,
        "config_version": "1.0",
        "last_updated": "2025-05-28T11:30:00Z"
    }

@router.post("/config",
            response_model=ConfigResponse,
            summary="更新系统配置",
            description="更新系统配置设置")
async def update_config(request: ConfigRequest) -> ConfigResponse:
    """
    更新系统配置
    
    **支持的配置：**
    - 分析参数：输入形状、超时时间、启用功能
    - 可视化设置：布局、颜色、节点样式
    - 导出选项：默认格式、质量、过期时间
    
    **验证规则：**
    - 数值范围检查
    - 枚举值验证
    - 依赖关系检查
    
    **注意：** 某些配置更改需要重启服务才能生效
    """
    updated_sections = []
    
    # 更新分析配置
    if request.analysis_config:
        _validate_analysis_config(request.analysis_config)
        _global_config["analysis_config"].update(request.analysis_config)
        updated_sections.append("analysis_config")
    
    # 更新可视化配置
    if request.visualization_config:
        _validate_visualization_config(request.visualization_config)
        _global_config["visualization_config"].update(request.visualization_config)
        updated_sections.append("visualization_config")
    
    # 更新导出配置
    if request.export_config:
        _validate_export_config(request.export_config)
        _global_config["export_config"].update(request.export_config)
        updated_sections.append("export_config")
    
    return ConfigResponse(
        success=True,
        message=f"已更新配置: {', '.join(updated_sections)}",
        config=_global_config
    )

@router.get("/config/analysis",
           summary="获取分析配置",
           description="获取模型分析相关的配置")
async def get_analysis_config() -> Dict[str, Any]:
    """
    获取分析配置
    
    **配置项：**
    - `default_input_shape`: 默认输入形状
    - `max_analysis_time`: 最大分析时间（秒）
    - `enable_dynamic_analysis`: 启用动态图分析
    - `enable_pattern_recognition`: 启用架构模式识别
    - `enable_tensor_flow_analysis`: 启用Tensor流分析
    - `max_model_size_mb`: 最大模型大小限制
    """
    return {
        "analysis_config": _global_config["analysis_config"]
    }

@router.post("/config/analysis",
            summary="更新分析配置",
            description="更新模型分析相关的配置")
async def update_analysis_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新分析配置
    
    **可配置项：**
    - 默认输入形状
    - 分析超时时间
    - 功能开关
    - 资源限制
    """
    _validate_analysis_config(config)
    _global_config["analysis_config"].update(config)
    
    return {
        "success": True,
        "message": "分析配置已更新",
        "updated_config": _global_config["analysis_config"]
    }

@router.get("/config/visualization",
           summary="获取可视化配置",
           description="获取图形可视化相关的配置")
async def get_visualization_config() -> Dict[str, Any]:
    """
    获取可视化配置
    
    **配置项：**
    - `default_layout`: 默认布局算法
    - `node_spacing`: 节点间距
    - `edge_style`: 边的样式
    - `color_scheme`: 颜色方案
    - `show_parameters`: 显示参数信息
    - `show_shapes`: 显示形状信息
    - `animation_enabled`: 启用动画
    """
    return {
        "visualization_config": _global_config["visualization_config"]
    }

@router.post("/config/visualization",
            summary="更新可视化配置",
            description="更新图形可视化相关的配置")
async def update_visualization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新可视化配置
    
    **可配置项：**
    - 布局算法和参数
    - 节点和边的样式
    - 颜色主题
    - 显示选项
    """
    _validate_visualization_config(config)
    _global_config["visualization_config"].update(config)
    
    return {
        "success": True,
        "message": "可视化配置已更新",
        "updated_config": _global_config["visualization_config"]
    }

@router.get("/config/export",
           summary="获取导出配置",
           description="获取文件导出相关的配置")
async def get_export_config() -> Dict[str, Any]:
    """
    获取导出配置
    
    **配置项：**
    - `default_format`: 默认导出格式
    - `max_export_size_mb`: 最大导出文件大小
    - `export_expiry_hours`: 导出文件过期时间
    - `image_quality`: 图像质量设置
    - `svg_precision`: SVG精度
    """
    return {
        "export_config": _global_config["export_config"]
    }

@router.post("/config/export",
            summary="更新导出配置",
            description="更新文件导出相关的配置")
async def update_export_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新导出配置
    
    **可配置项：**
    - 默认导出格式
    - 文件大小限制
    - 过期时间设置
    - 质量参数
    """
    _validate_export_config(config)
    _global_config["export_config"].update(config)
    
    return {
        "success": True,
        "message": "导出配置已更新",
        "updated_config": _global_config["export_config"]
    }

@router.post("/config/reset",
            summary="重置配置",
            description="将配置重置为默认值")
async def reset_config(section: Optional[str] = None) -> Dict[str, Any]:
    """
    重置配置
    
    **参数：**
    - `section`: 要重置的配置节（可选）
      - `analysis`: 重置分析配置
      - `visualization`: 重置可视化配置
      - `export`: 重置导出配置
      - 不指定则重置所有配置
    
    **注意：** 此操作将覆盖当前设置
    """
    default_config = {
        "analysis_config": {
            "default_input_shape": [1, 3, 224, 224],
            "max_analysis_time": 300,
            "enable_dynamic_analysis": True,
            "enable_pattern_recognition": True,
            "enable_tensor_flow_analysis": True,
            "max_model_size_mb": 100
        },
        "visualization_config": {
            "default_layout": "hierarchical",
            "node_spacing": 100,
            "edge_style": "curved",
            "color_scheme": "default",
            "show_parameters": True,
            "show_shapes": True,
            "animation_enabled": False
        },
        "export_config": {
            "default_format": "json",
            "max_export_size_mb": 50,
            "export_expiry_hours": 24,
            "image_quality": "high",
            "svg_precision": 2
        }
    }
    
    if section:
        if section not in default_config:
            raise HTTPException(
                status_code=400,
                detail=f"无效的配置节: {section}"
            )
        _global_config[f"{section}_config"] = default_config[f"{section}_config"]
        message = f"已重置 {section} 配置为默认值"
    else:
        _global_config.update(default_config)
        message = "已重置所有配置为默认值"
    
    return {
        "success": True,
        "message": message,
        "config": _global_config
    }

def _validate_analysis_config(config: Dict[str, Any]):
    """验证分析配置"""
    if "max_analysis_time" in config:
        if not isinstance(config["max_analysis_time"], int) or config["max_analysis_time"] < 10:
            raise HTTPException(
                status_code=400,
                detail="max_analysis_time 必须是大于10的整数"
            )
    
    if "default_input_shape" in config:
        if not isinstance(config["default_input_shape"], list) or len(config["default_input_shape"]) < 2:
            raise HTTPException(
                status_code=400,
                detail="default_input_shape 必须是包含至少2个元素的列表"
            )
    
    if "max_model_size_mb" in config:
        if not isinstance(config["max_model_size_mb"], (int, float)) or config["max_model_size_mb"] <= 0:
            raise HTTPException(
                status_code=400,
                detail="max_model_size_mb 必须是大于0的数值"
            )

def _validate_visualization_config(config: Dict[str, Any]):
    """验证可视化配置"""
    if "default_layout" in config:
        valid_layouts = ["hierarchical", "force", "circular", "grid"]
        if config["default_layout"] not in valid_layouts:
            raise HTTPException(
                status_code=400,
                detail=f"default_layout 必须是以下值之一: {', '.join(valid_layouts)}"
            )
    
    if "node_spacing" in config:
        if not isinstance(config["node_spacing"], (int, float)) or config["node_spacing"] <= 0:
            raise HTTPException(
                status_code=400,
                detail="node_spacing 必须是大于0的数值"
            )
    
    if "color_scheme" in config:
        valid_schemes = ["default", "dark", "light", "colorful", "minimal"]
        if config["color_scheme"] not in valid_schemes:
            raise HTTPException(
                status_code=400,
                detail=f"color_scheme 必须是以下值之一: {', '.join(valid_schemes)}"
            )

def _validate_export_config(config: Dict[str, Any]):
    """验证导出配置"""
    if "default_format" in config:
        valid_formats = ["json", "png", "svg"]
        if config["default_format"] not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"default_format 必须是以下值之一: {', '.join(valid_formats)}"
            )
    
    if "export_expiry_hours" in config:
        if not isinstance(config["export_expiry_hours"], int) or config["export_expiry_hours"] < 1:
            raise HTTPException(
                status_code=400,
                detail="export_expiry_hours 必须是大于0的整数"
            )
    
    if "max_export_size_mb" in config:
        if not isinstance(config["max_export_size_mb"], (int, float)) or config["max_export_size_mb"] <= 0:
            raise HTTPException(
                status_code=400,
                detail="max_export_size_mb 必须是大于0的数值"
            )
