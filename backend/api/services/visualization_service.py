"""
可视化服务
处理可视化数据生成和管理
"""

from typing import Dict, Any, Optional
from .model_service import model_analysis_service

class VisualizationService:
    """可视化服务类"""
    
    def __init__(self):
        self.model_service = model_analysis_service
    
    def get_visualization_data(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取可视化数据"""
        # 获取任务结果
        result = self.model_service.get_task_result(task_id)
        if not result:
            return None
        
        # 提取可视化数据
        viz_data = result.get("visualization_data")
        if not viz_data:
            return None
        
        # 准备响应数据
        enhanced_info = result.get("enhanced_info")
        
        response_data = {
            "task_id": task_id,
            "model_name": enhanced_info.get("model_name", "Unknown"),
            "graph_type": enhanced_info.get("network_graph", {}).get("graph_type", "unknown"),
            "nodes": viz_data.get("nodes", []),
            "edges": viz_data.get("edges", []),
            "metadata": viz_data.get("metadata", {}),
            "statistics": self._calculate_statistics(viz_data)
        }
        
        # 添加增强分析数据
        if "dynamic_info" in viz_data:
            response_data["dynamic_info"] = viz_data["dynamic_info"]
            
        if "architecture_patterns" in viz_data:
            response_data["architecture_patterns"] = viz_data["architecture_patterns"]
            
        if "tensor_flow" in viz_data:
            response_data["tensor_flow"] = viz_data["tensor_flow"]
        
        return response_data
    
    def _calculate_statistics(self, viz_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计信息"""
        nodes = viz_data.get("nodes", [])
        edges = viz_data.get("edges", [])
        
        # 基础统计
        stats = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "layer_types": {},
            "connection_types": {}
        }
        
        # 统计层类型
        for node in nodes:
            node_type = node.get("type", "unknown")
            stats["layer_types"][node_type] = stats["layer_types"].get(node_type, 0) + 1
        
        # 统计连接类型
        for edge in edges:
            conn_type = edge.get("connection_type", "unknown")
            stats["connection_types"][conn_type] = stats["connection_types"].get(conn_type, 0) + 1
        
        # 参数统计
        total_params = 0
        trainable_params = 0
        
        for node in nodes:
            if "num_parameters" in node:
                total_params += node.get("num_parameters", 0)
                if node.get("trainable", True):
                    trainable_params += node.get("num_parameters", 0)
        
        stats["total_parameters"] = total_params
        stats["trainable_parameters"] = trainable_params
        
        # 增强统计
        if "dynamic_info" in viz_data:
            dynamic_info = viz_data["dynamic_info"]
            stats["has_dynamic_control"] = dynamic_info.get("is_dynamic", False)
            stats["execution_paths"] = len(dynamic_info.get("execution_paths", []))
        
        if "architecture_patterns" in viz_data:
            arch_info = viz_data["architecture_patterns"]
            stats["architecture_type"] = arch_info.get("architecture_type", "unknown")
            stats["residual_connections"] = len(arch_info.get("residual_connections", []))
            stats["attention_patterns"] = len(arch_info.get("attention_patterns", []))
        
        if "tensor_flow" in viz_data:
            tensor_info = viz_data["tensor_flow"]
            stats["tensor_operations"] = len(tensor_info.get("tensor_operations", []))
            stats["data_flow_paths"] = len(tensor_info.get("data_flow_paths", []))
        
        return stats
    
    def get_node_details(self, task_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点详细信息"""
        viz_data = self.get_visualization_data(task_id)
        if not viz_data:
            return None
        
        # 查找指定节点
        for node in viz_data.get("nodes", []):
            if node.get("id") == node_id:
                return {
                    "node": node,
                    "connections": self._get_node_connections(node_id, viz_data.get("edges", []))
                }
        
        return None
    
    def _get_node_connections(self, node_id: str, edges: list) -> Dict[str, list]:
        """获取节点连接信息"""
        incoming = []
        outgoing = []
        
        for edge in edges:
            if edge.get("target") == node_id:
                incoming.append(edge)
            elif edge.get("source") == node_id:
                outgoing.append(edge)
        
        return {
            "incoming": incoming,
            "outgoing": outgoing
        }
    
    def get_subgraph(self, task_id: str, node_ids: list) -> Optional[Dict[str, Any]]:
        """获取子图"""
        viz_data = self.get_visualization_data(task_id)
        if not viz_data:
            return None
        
        # 过滤节点
        filtered_nodes = [
            node for node in viz_data.get("nodes", [])
            if node.get("id") in node_ids
        ]
        
        # 过滤边（只保留两端都在子图中的边）
        filtered_edges = [
            edge for edge in viz_data.get("edges", [])
            if edge.get("source") in node_ids and edge.get("target") in node_ids
        ]
        
        return {
            "task_id": task_id,
            "subgraph": {
                "nodes": filtered_nodes,
                "edges": filtered_edges,
                "statistics": {
                    "node_count": len(filtered_nodes),
                    "edge_count": len(filtered_edges)
                }
            }
        }

# 创建单例服务实例
visualization_service = VisualizationService()
