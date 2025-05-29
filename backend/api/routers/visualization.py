"""
可视化API路由
处理可视化数据相关的请求
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.api.schemas.response import VisualizationResponse
from backend.api.services.enhanced_visualization_service_complete import enhanced_visualization_service
from backend.api.middleware.error_handler import TaskNotFoundError

router = APIRouter()

@router.get("/models/{task_id}/visualization",
           response_model=VisualizationResponse,
           summary="获取可视化数据",
           description="获取已完成分析任务的可视化数据")
async def get_visualization_data(task_id: str) -> VisualizationResponse:
    """
    获取可视化数据
    
    **包含数据：**
    - 节点信息（层结构、参数、形状）
    - 边信息（连接关系、数据流）
    - 元数据（模型统计、复杂度）
    - 增强分析数据（动态图、架构模式、Tensor流）
    
    **数据格式：**
    - 节点：ID、类型、标签、参数、位置坐标
    - 边：源节点、目标节点、连接类型、权重
    - 统计：参数总数、层数分布、连接复杂度
    """
    viz_data = enhanced_visualization_service.get_visualization_data(task_id)
    
    if not viz_data:
        raise TaskNotFoundError(task_id)
    
    return VisualizationResponse(**viz_data)

@router.get("/models/{task_id}/nodes/{node_id}",
           summary="获取节点详细信息",
           description="获取指定节点的详细信息和连接关系")
async def get_node_details(task_id: str, node_id: str) -> Dict[str, Any]:
    """
    获取节点详细信息
    
    **返回信息：**
    - 节点完整属性
    - 输入连接列表
    - 输出连接列表
    - 参数详情
    - 形状变化信息
    """
    node_details = enhanced_visualization_service.get_node_details(task_id, node_id)
    
    if not node_details:
        raise HTTPException(
            status_code=404,
            detail=f"节点 {node_id} 在任务 {task_id} 中未找到"
        )
    
    return {
        "task_id": task_id,
        "node_id": node_id,
        "details": node_details
    }

@router.post("/models/{task_id}/subgraph",
            summary="获取子图",
            description="根据指定的节点ID列表获取子图")
async def get_subgraph(task_id: str, node_ids: List[str]) -> Dict[str, Any]:
    """
    获取子图
    
    **用途：**
    - 分析特定层组合
    - 提取关键路径
    - 局部结构可视化
    
    **参数：**
    - `node_ids`: 要包含在子图中的节点ID列表
    
    **返回：**
    - 过滤后的节点和边
    - 子图统计信息
    """
    if not node_ids:
        raise HTTPException(
            status_code=400,
            detail="至少需要指定一个节点ID"
        )
    
    subgraph_data = enhanced_visualization_service.get_subgraph(task_id, node_ids)
    
    if not subgraph_data:
        raise TaskNotFoundError(task_id)
    
    return subgraph_data

@router.get("/models/{task_id}/statistics",
           summary="获取图统计信息",
           description="获取网络图的详细统计信息")
async def get_graph_statistics(task_id: str) -> Dict[str, Any]:
    """
    获取图统计信息
    
    **统计内容：**
    - 基础统计：节点数、边数、参数总数
    - 层类型分布：各类型层的数量
    - 连接分析：连接类型、复杂度评分
    - 增强统计：动态特性、架构模式
    """
    viz_data = enhanced_visualization_service.get_visualization_data(task_id)
    
    if not viz_data:
        raise TaskNotFoundError(task_id)
    
    # 提取统计信息
    statistics = viz_data.get("statistics", {})
    
    # 添加额外的统计分析
    enhanced_stats = {
        "basic_stats": {
            "node_count": statistics.get("node_count", 0),
            "edge_count": statistics.get("edge_count", 0),
            "total_parameters": statistics.get("total_parameters", 0),
            "trainable_parameters": statistics.get("trainable_parameters", 0)
        },
        "layer_distribution": statistics.get("layer_types", {}),
        "connection_analysis": {
            "connection_types": statistics.get("connection_types", {}),
            "average_connections_per_node": (
                statistics.get("edge_count", 0) / max(statistics.get("node_count", 1), 1)
            )
        },
        "enhanced_features": {
            "has_dynamic_control": statistics.get("has_dynamic_control", False),
            "architecture_type": statistics.get("architecture_type", "unknown"),
            "execution_paths": statistics.get("execution_paths", 0),
            "residual_connections": statistics.get("residual_connections", 0),
            "tensor_operations": statistics.get("tensor_operations", 0)
        }
    }
    
    return {
        "task_id": task_id,
        "statistics": enhanced_stats
    }

@router.get("/models/{task_id}/layout",
           summary="获取布局信息",
           description="获取图形布局的坐标和样式信息")
async def get_layout_data(task_id: str, layout_type: Optional[str] = "hierarchical") -> Dict[str, Any]:
    """
    获取布局信息
    
    **布局类型：**
    - `hierarchical`: 层次布局（默认）
    - `force`: 力导向布局
    - `circular`: 环形布局
    - `grid`: 网格布局
    
    **返回数据：**
    - 节点位置坐标
    - 边路径信息
    - 布局参数
    - 视窗设置
    """
    viz_data = enhanced_visualization_service.get_visualization_data(task_id)
    
    if not viz_data:
        raise TaskNotFoundError(task_id)
    
    nodes = viz_data.get("nodes", [])
    edges = viz_data.get("edges", [])
    
    # 根据布局类型计算坐标
    layout_data = _calculate_layout(nodes, edges, layout_type)
    
    return {
        "task_id": task_id,
        "layout_type": layout_type,
        "layout_data": layout_data,
        "viewport": {
            "width": layout_data.get("canvas_width", 1200),
            "height": layout_data.get("canvas_height", 800),
            "zoom_level": 1.0
        }
    }

def _calculate_layout(nodes: List[Dict], edges: List[Dict], layout_type: str) -> Dict[str, Any]:
    """计算布局坐标"""
    
    if layout_type == "hierarchical":
        return _hierarchical_layout(nodes, edges)
    elif layout_type == "circular":
        return _circular_layout(nodes)
    elif layout_type == "grid":
        return _grid_layout(nodes)
    else:
        # 默认层次布局
        return _hierarchical_layout(nodes, edges)

def _hierarchical_layout(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
    """层次布局算法"""
    node_positions = {}
    canvas_width = 1200
    canvas_height = 800
    
    # 简单的层次布局
    for i, node in enumerate(nodes):
        # 根据节点类型确定层级
        layer = _get_node_layer(node)
        
        # 计算位置
        x = (layer + 1) * (canvas_width // 6)
        y = (i % 5 + 1) * (canvas_height // 6)
        
        node_positions[node["id"]] = {"x": x, "y": y}
    
    return {
        "node_positions": node_positions,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "layout_algorithm": "hierarchical"
    }

def _circular_layout(nodes: List[Dict]) -> Dict[str, Any]:
    """环形布局算法"""
    import math
    
    node_positions = {}
    canvas_width = 800
    canvas_height = 800
    center_x, center_y = canvas_width // 2, canvas_height // 2
    radius = min(canvas_width, canvas_height) // 3
    
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / len(nodes)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        
        node_positions[node["id"]] = {"x": int(x), "y": int(y)}
    
    return {
        "node_positions": node_positions,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "layout_algorithm": "circular"
    }

def _grid_layout(nodes: List[Dict]) -> Dict[str, Any]:
    """网格布局算法"""
    import math
    
    node_positions = {}
    canvas_width = 1000
    canvas_height = 800
    
    cols = int(math.sqrt(len(nodes))) + 1
    cell_width = canvas_width // cols
    cell_height = canvas_height // ((len(nodes) // cols) + 1)
    
    for i, node in enumerate(nodes):
        row = i // cols
        col = i % cols
        
        x = (col + 1) * cell_width
        y = (row + 1) * cell_height
        
        node_positions[node["id"]] = {"x": x, "y": y}
    
    return {
        "node_positions": node_positions,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "layout_algorithm": "grid"
    }

def _get_node_layer(node: Dict) -> int:
    """根据节点类型确定层级"""
    node_type = node.get("type", "")
    
    # 输入层
    if "input" in node_type.lower() or node_type in ["Input"]:
        return 0
    # 卷积层
    elif "conv" in node_type.lower():
        return 1
    # 池化层
    elif "pool" in node_type.lower():
        return 2
    # 全连接层
    elif "linear" in node_type.lower() or "fc" in node_type.lower():
        return 4
    # 输出层
    elif "output" in node_type.lower():
        return 5
    # 其他层
    else:
        return 3
