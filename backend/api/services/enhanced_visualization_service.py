"""
增强可视化服务
提供更丰富的模型结构图数据
"""

from typing import Dict, Any, Optional, List
import math
from .model_service import model_analysis_service

class EnhancedVisualizationService:
    """增强可视化服务类"""
    
    def __init__(self):
        self.model_service = model_analysis_service
    
    def get_visualization_data(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取增强可视化数据"""
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
        
        # 增强节点数据
        enhanced_nodes = self._enhance_nodes_data(viz_data.get("nodes", []), enhanced_info)
        
        # 增强边数据
        enhanced_edges = self._enhance_edges_data(viz_data.get("edges", []), enhanced_info)
        
        response_data = {
            "task_id": task_id,
            "model_name": enhanced_info.get("model_name", "Unknown"),
            "graph_type": enhanced_info.get("network_graph", {}).get("graph_type", "unknown"),
            "nodes": enhanced_nodes,
            "edges": enhanced_edges,
            "metadata": self._build_enhanced_metadata(viz_data, enhanced_info),
            "statistics": self._calculate_enhanced_statistics(viz_data, enhanced_info)
        }
        
        # 添加增强分析数据
        if "dynamic_info" in viz_data:
            response_data["dynamic_info"] = viz_data["dynamic_info"]
            
        if "architecture_patterns" in viz_data:
            response_data["architecture_patterns"] = viz_data["architecture_patterns"]
            
        if "tensor_flow" in viz_data:
            response_data["tensor_flow"] = viz_data["tensor_flow"]
        
        return response_data
    
    def _enhance_nodes_data(self, nodes: List[Dict], enhanced_info: Dict) -> List[Dict]:
        """增强节点数据"""
        enhanced_nodes = []
        
        # 获取结构信息和复杂度分析
        structure_info = enhanced_info.get("structure_info", {})
        complexity = enhanced_info.get("complexity_analysis", {})
        
        for node in nodes:
            enhanced_node = dict(node)  # 复制原始节点数据
            
            # 增强基本信息
            enhanced_node.update({
                "category": self._categorize_layer_type(node.get("type", "")),
                "importance_score": self._calculate_importance_score(node),
                "computational_cost": self._estimate_computational_cost(node),
                "memory_footprint": self._estimate_memory_usage(node)
            })
            
            # 添加详细配置参数
            if "parameters" in node and isinstance(node["parameters"], dict):
                enhanced_node["detailed_config"] = self._format_layer_config(
                    node["type"], node["parameters"]
                )
            
            # 添加性能指标
            enhanced_node["performance_metrics"] = self._calculate_performance_metrics(node)
            
            # 添加可视化提示
            enhanced_node["visual_hints"] = self._generate_visual_hints(node)
            
            enhanced_nodes.append(enhanced_node)
        
        return enhanced_nodes
    
    def _enhance_edges_data(self, edges: List[Dict], enhanced_info: Dict) -> List[Dict]:
        """增强边数据"""
        enhanced_edges = []
        
        for edge in edges:
            enhanced_edge = dict(edge)  # 复制原始边数据
            
            # 增强连接信息
            enhanced_edge.update({
                "data_bandwidth": self._estimate_data_bandwidth(edge),
                "tensor_info": self._extract_tensor_info(edge),
                "connection_strength": self._calculate_connection_strength(edge),
                "latency_impact": self._estimate_latency_impact(edge)
            })
            
            # 添加视觉样式提示
            enhanced_edge["visual_style"] = self._generate_edge_style(edge)
            
            enhanced_edges.append(enhanced_edge)
        
        return enhanced_edges
    
    def _build_enhanced_metadata(self, viz_data: Dict, enhanced_info: Dict) -> Dict[str, Any]:
        """构建增强元数据"""
        metadata = viz_data.get("metadata", {}).copy()
        
        # 添加架构信息
        if "architecture_patterns" in enhanced_info:
            arch_patterns = enhanced_info["architecture_patterns"]
            metadata["architecture_info"] = {
                "pattern_type": arch_patterns.get("pattern_type", "unknown"),
                "has_residual": len(arch_patterns.get("residual_connections", [])) > 0,
                "has_attention": len(arch_patterns.get("attention_patterns", [])) > 0,
                "has_branching": len(arch_patterns.get("branch_points", [])) > 0
            }
        
        # 添加复杂度信息
        if "complexity_analysis" in enhanced_info:
            complexity = enhanced_info["complexity_analysis"]
            metadata["complexity"] = complexity
        
        # 添加推荐布局
        metadata["recommended_layout"] = self._recommend_layout(enhanced_info)
        
        # 添加可视化配置
        metadata["visualization_config"] = {
            "optimal_zoom": self._calculate_optimal_zoom(viz_data),
            "focus_nodes": self._identify_focus_nodes(viz_data),
            "color_scheme": self._recommend_color_scheme(enhanced_info)
        }
        
        return metadata
    
    def _calculate_enhanced_statistics(self, viz_data: Dict, enhanced_info: Dict) -> Dict[str, Any]:
        """计算增强统计信息"""
        nodes = viz_data.get("nodes", [])
        edges = viz_data.get("edges", [])
        
        # 基础统计
        stats = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "layer_types": {},
            "connection_types": {},
            "graph_metrics": self._calculate_graph_metrics(nodes, edges)
        }
        
        # 统计层类型
        for node in nodes:
            node_type = node.get("type", "unknown")
            category = self._categorize_layer_type(node_type)
            stats["layer_types"][category] = stats["layer_types"].get(category, 0) + 1
        
        # 统计连接类型
        for edge in edges:
            conn_type = edge.get("connection_type", "unknown")
            stats["connection_types"][conn_type] = stats["connection_types"].get(conn_type, 0) + 1
        
        # 参数统计
        total_params = 0
        trainable_params = 0
        total_flops = 0
        total_memory = 0
        
        for node in nodes:
            if "num_parameters" in node:
                params = node.get("num_parameters", 0)
                total_params += params
                if node.get("trainable", True):
                    trainable_params += params
            
            if "flops" in node:
                total_flops += node.get("flops", 0)
            
            if "memory_usage" in node:
                total_memory += node.get("memory_usage", 0)
        
        stats.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "total_flops": total_flops,
            "total_memory_mb": total_memory / (1024 * 1024) if total_memory > 0 else 0
        })
        
        # 从结构信息更新统计
        if "structure_info" in enhanced_info:
            structure = enhanced_info["structure_info"]
            if hasattr(structure, 'total_parameters'):
                stats["total_parameters"] = structure.total_parameters
            if hasattr(structure, 'trainable_parameters'):
                stats["trainable_parameters"] = structure.trainable_parameters
            if hasattr(structure, 'model_size_mb'):
                stats["model_size_mb"] = structure.model_size_mb
        
        # 增强统计
        if "dynamic_analysis" in enhanced_info:
            dynamic_info = enhanced_info["dynamic_analysis"]
            if hasattr(dynamic_info, 'has_dynamic_control'):
                stats["has_dynamic_control"] = dynamic_info.has_dynamic_control
            if hasattr(dynamic_info, 'execution_paths'):
                stats["execution_paths"] = len(dynamic_info.execution_paths)
        
        return stats
    
    def _categorize_layer_type(self, layer_type: str) -> str:
        """分类层类型"""
        type_lower = layer_type.lower()
        
        if any(x in type_lower for x in ['conv', 'convolution']):
            return 'convolution'
        elif any(x in type_lower for x in ['linear', 'dense', 'fc', 'fully']):
            return 'dense'
        elif any(x in type_lower for x in ['pool', 'pooling']):
            return 'pooling'
        elif any(x in type_lower for x in ['relu', 'sigmoid', 'tanh', 'activation', 'gelu', 'swish']):
            return 'activation'
        elif any(x in type_lower for x in ['batch', 'layer', 'norm', 'normalization']):
            return 'normalization'
        elif 'dropout' in type_lower:
            return 'regularization'
        elif any(x in type_lower for x in ['attention', 'self_attention', 'multihead']):
            return 'attention'
        elif any(x in type_lower for x in ['embedding', 'embed']):
            return 'embedding'
        elif any(x in type_lower for x in ['rnn', 'lstm', 'gru', 'recurrent']):
            return 'recurrent'
        else:
            return 'other'
    
    def _calculate_importance_score(self, node: Dict) -> float:
        """计算节点重要性分数"""
        score = 0.0
        
        # 基于参数数量
        params = node.get("num_parameters", 0)
        if params > 0:
            score += min(math.log10(params + 1) / 10, 1.0)
        
        # 基于层类型
        layer_type = node.get("type", "").lower()
        if any(x in layer_type for x in ['conv', 'linear', 'attention']):
            score += 0.3
        elif any(x in layer_type for x in ['norm', 'activation']):
            score += 0.1
        
        # 基于计算量
        flops = node.get("flops", 0)
        if flops > 0:
            score += min(math.log10(flops + 1) / 15, 0.5)
        
        return min(score, 1.0)
    
    def _estimate_computational_cost(self, node: Dict) -> Dict[str, Any]:
        """估算计算成本"""
        layer_type = node.get("type", "").lower()
        params = node.get("num_parameters", 0)
        
        # 基于层类型的相对计算成本
        cost_multipliers = {
            'conv': 3.0,
            'linear': 2.0,
            'attention': 4.0,
            'norm': 0.5,
            'activation': 0.2,
            'pool': 0.3
        }
        
        base_cost = 1.0
        for key, multiplier in cost_multipliers.items():
            if key in layer_type:
                base_cost = multiplier
                break
        
        # 基于参数数量调整
        param_factor = math.log10(params + 1) / 10 if params > 0 else 0.1
        
        estimated_cost = base_cost * param_factor
        
        return {
            "relative_cost": estimated_cost,
            "cost_category": "high" if estimated_cost > 2.0 else "medium" if estimated_cost > 0.5 else "low",
            "flops_estimate": node.get("flops", 0)
        }
    
    def _estimate_memory_usage(self, node: Dict) -> Dict[str, Any]:
        """估算内存使用"""
        params = node.get("num_parameters", 0)
        
        # 参数内存 (假设 float32, 每个参数4字节)
        param_memory = params * 4
        
        # 激活内存 (基于输出形状估算)
        activation_memory = 0
        if "output_shape" in node:
            output_shape = node["output_shape"]
            if isinstance(output_shape, (list, tuple)) and len(output_shape) > 0:
                try:
                    activation_size = 1
                    for dim in output_shape:
                        if isinstance(dim, int) and dim > 0:
                            activation_size *= dim
                    activation_memory = activation_size * 4  # float32
                except:
                    activation_memory = 0
        
        total_memory = param_memory + activation_memory
        
        return {
            "parameter_memory_bytes": param_memory,
            "activation_memory_bytes": activation_memory,
            "total_memory_bytes": total_memory,
            "memory_mb": total_memory / (1024 * 1024) if total_memory > 0 else 0
        }
    
    def _format_layer_config(self, layer_type: str, parameters: Dict) -> Dict[str, Any]:
        """格式化层配置参数"""
        # 根据层类型格式化关键参数
        formatted_config = {}
        
        type_lower = layer_type.lower()
        
        if 'conv' in type_lower:
            formatted_config.update({
                "kernel_size": parameters.get("kernel_size"),
                "stride": parameters.get("stride", 1),
                "padding": parameters.get("padding", 0),
                "dilation": parameters.get("dilation", 1),
                "groups": parameters.get("groups", 1),
                "bias": parameters.get("bias", True)
            })
        elif 'linear' in type_lower:
            formatted_config.update({
                "in_features": parameters.get("in_features"),
                "out_features": parameters.get("out_features"),
                "bias": parameters.get("bias", True)
            })
        elif 'pool' in type_lower:
            formatted_config.update({
                "kernel_size": parameters.get("kernel_size"),
                "stride": parameters.get("stride"),
                "padding": parameters.get("padding", 0)
            })
        elif 'norm' in type_lower:
            formatted_config.update({
                "num_features": parameters.get("num_features"),
                "eps": parameters.get("eps", 1e-5),
                "momentum": parameters.get("momentum", 0.1),
                "affine": parameters.get("affine", True)
            })
        
        # 移除None值
        return {k: v for k, v in formatted_config.items() if v is not None}
    
    def _calculate_performance_metrics(self, node: Dict) -> Dict[str, Any]:
        """计算性能指标"""
        return {
            "efficiency_score": self._calculate_efficiency_score(node),
            "bottleneck_risk": self._assess_bottleneck_risk(node),
            "parallelization_potential": self._assess_parallelization_potential(node)
        }
    
    def _calculate_efficiency_score(self, node: Dict) -> float:
        """计算效率分数"""
        params = node.get("num_parameters", 0)
        flops = node.get("flops", 0)
        
        if params == 0:
            return 1.0  # 无参数层效率高
        
        # FLOPS per parameter ratio
        if flops > 0:
            efficiency = flops / params
            return min(efficiency / 1000, 1.0)  # 归一化
        
        return 0.5  # 默认中等效率
    
    def _assess_bottleneck_risk(self, node: Dict) -> str:
        """评估瓶颈风险"""
        cost = self._estimate_computational_cost(node)
        relative_cost = cost["relative_cost"]
        
        if relative_cost > 3.0:
            return "high"
        elif relative_cost > 1.0:
            return "medium"
        else:
            return "low"
    
    def _assess_parallelization_potential(self, node: Dict) -> str:
        """评估并行化潜力"""
        layer_type = node.get("type", "").lower()
        
        if any(x in layer_type for x in ['conv', 'linear']):
            return "high"
        elif any(x in layer_type for x in ['norm', 'pool']):
            return "medium"
        else:
            return "low"
    
    def _generate_visual_hints(self, node: Dict) -> Dict[str, Any]:
        """生成可视化提示"""
        return {
            "suggested_size": self._suggest_node_size(node),
            "suggested_color": self._suggest_node_color(node),
            "suggested_shape": self._suggest_node_shape(node),
            "highlight_priority": self._calculate_highlight_priority(node)
        }
    
    def _suggest_node_size(self, node: Dict) -> str:
        """建议节点大小"""
        importance = self._calculate_importance_score(node)
        
        if importance > 0.7:
            return "large"
        elif importance > 0.3:
            return "medium"
        else:
            return "small"
    
    def _suggest_node_color(self, node: Dict) -> str:
        """建议节点颜色"""
        category = self._categorize_layer_type(node.get("type", ""))
        
        color_map = {
            'convolution': 'red',
            'dense': 'blue',
            'pooling': 'cyan',
            'activation': 'green',
            'normalization': 'yellow',
            'regularization': 'purple',
            'attention': 'orange',
            'embedding': 'pink',
            'recurrent': 'brown'
        }
        
        return color_map.get(category, 'gray')
    
    def _suggest_node_shape(self, node: Dict) -> str:
        """建议节点形状"""
        category = self._categorize_layer_type(node.get("type", ""))
        
        shape_map = {
            'convolution': 'box',
            'dense': 'ellipse',
            'pooling': 'triangle',
            'activation': 'diamond',
            'normalization': 'square',
            'regularization': 'dot',
            'attention': 'star',
            'embedding': 'box',
            'recurrent': 'ellipse'
        }
        
        return shape_map.get(category, 'box')
    
    def _calculate_highlight_priority(self, node: Dict) -> int:
        """计算高亮优先级"""
        importance = self._calculate_importance_score(node)
        
        if importance > 0.8:
            return 3  # 最高优先级
        elif importance > 0.5:
            return 2  # 中等优先级
        elif importance > 0.2:
            return 1  # 低优先级
        else:
            return 0  # 不高亮
    
    def _estimate_data_bandwidth(self, edge: Dict) -> Dict[str, Any]:
        """估算数据带宽"""
        # 基于tensor形状估算数据大小
        tensor_shape = edge.get("tensor_shape")
        if tensor_shape and isinstance(tensor_shape, (list, tuple)):
            try:
                tensor_size = 1
                for dim in tensor_shape:
                    if isinstance(dim, int) and dim > 0:
                        tensor_size *= dim
                data_bytes = tensor_size * 4  # float32
                
                return {
                    "tensor_size": tensor_size,
                    "data_bytes": data_bytes,
                    "data_mb": data_bytes / (1024 * 1024),
                    "bandwidth_category": "high" if data_bytes > 1024*1024 else "medium" if data_bytes > 1024 else "low"
                }
            except:
                pass
        
        return {
            "tensor_size": 0,
            "data_bytes": 0,
            "data_mb": 0,
            "bandwidth_category": "unknown"
        }
    
    def _extract_tensor_info(self, edge: Dict) -> Dict[str, Any]:
        """提取tensor信息"""
        return {
            "shape": edge.get("tensor_shape"),
            "dtype": edge.get("tensor_dtype", "float32"),
            "requires_grad": edge.get("requires_grad", True),
            "memory_format": edge.get("memory_format", "contiguous")
        }
    
    def _calculate_connection_strength(self, edge: Dict) -> float:
        """计算连接强度"""
        conn_type = edge.get("connection_type", "sequential")
        
        # 基于连接类型的强度权重
        strength_weights = {
            "sequential": 1.0,
            "residual": 0.8,
            "attention": 0.9,
            "branch": 0.6,
            "merge": 0.7,
            "dense": 0.5
        }
        
        base_strength = strength_weights.get(conn_type, 0.5)
        
        # 基于数据带宽调整
        bandwidth = self._estimate_data_bandwidth(edge)
        bandwidth_factor = min(bandwidth["data_mb"] / 10, 1.0) if bandwidth["data_mb"] > 0 else 0.5
        
        return min(base_strength * (1 + bandwidth_factor), 1.0)
    
    def _estimate_latency_impact(self, edge: Dict) -> str:
        """估算延迟影响"""
        bandwidth = self._estimate_data_bandwidth(edge)
        
        if bandwidth["data_mb"] > 100:
            return "high"
        elif bandwidth["data_mb"] > 10:
            return "medium"
        else:
            return "low"
    
    def _generate_edge_style(self, edge: Dict) -> Dict[str, Any]:
        """生成边的视觉样式"""
        conn_type = edge.get("connection_type", "sequential")
        strength = self._calculate_connection_strength(edge)
        
        return {
            "suggested_width": max(1, int(strength * 5)),
            "suggested_opacity": max(0.3, strength),
            "animation_speed": "fast" if strength > 0.8 else "medium" if strength > 0.5 else "slow",
            "dash_pattern": self._get_dash_pattern(conn_type)
        }
    
    def _get_dash_pattern(self, connection_type: str) -> List[int]:
        """获取虚线模式"""
        patterns = {
            "sequential": [],
            "residual": [5, 5],
            "attention": [10, 5],
            "branch": [15, 5, 5, 5],
            "merge": [],
            "dense": [2, 3]
        }
        return patterns.get(connection_type, [])
    
    def _calculate_graph_metrics(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
        """计算图形指标"""
        node_count = len(nodes)
        edge_count = len(edges)
        
        # 计算图密度
        max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 1
        density = edge_count / max_edges if max_edges > 0 else 0
        
        # 计算平均度数
        avg_degree = (2 * edge_count) / node_count if node_count > 0 else 0
        
        return {
            "density": density,
            "average_degree": avg_degree,
            "complexity_score": density * avg_degree,
            "is_sparse": density < 0.1,
            "is_dense": density > 0.7
        }
    
    def _recommend_layout(self, enhanced_info: Dict) -> str:
        """推荐布局算法"""
        # 基于架构模式推荐布局
        if "architecture_patterns" in enhanced_info:
            arch_patterns = enhanced_info["architecture_patterns"]
            pattern_type = arch_patterns.get("pattern_type", "unknown")
            
            if pattern_type in ["transformer", "attention"]:
                return "network"
            elif pattern_type in ["resnet", "densenet"]:
                return "hierarchical"
            elif pattern_type == "complex":
                return "circular"
        
        # 基于图结构推荐
        if "complexity_analysis" in enhanced_info:
            complexity = enhanced_info["complexity_analysis"]
            if complexity.get("has_branching", False):
                return "network"
            elif complexity.get("has_residual", False):
                return "hierarchical"
        
        return "hierarchical"  # 默认布局
    
    def _calculate_optimal_zoom(self, viz_data: Dict) -> float:
        """计算最佳缩放级别"""
        node_count = len(viz_data.get("nodes", []))
        
        if node_count <= 10:
            return 1.0
        elif node_count <= 50:
            return 0.8
        elif node_count <= 100:
            return 0.6
        else:
            return 0.4
    
    def _identify_focus_nodes(self, viz_data: Dict) -> List[str]:
        """识别焦点节点"""
        nodes = viz_data.get("nodes", [])
        focus_nodes = []
        
        for node in nodes:
            importance = self._calculate_importance_score(node)
            if importance > 0.7:
                focus_nodes.append(node.get("id"))
        
        return focus_nodes[:5]  # 最多返回5个焦点节点
    
    def _recommend_color_scheme(self, enhanced_info: Dict) -> str:
        """推荐颜色方案"""
        if "architecture_patterns" in enhanced_info:
            arch_patterns = enhanced_info["architecture_patterns"]
            pattern_type = arch_patterns.get("pattern_type", "unknown")
            
            if pattern_type == "transformer":
                return "warm"
            elif pattern_type in ["resnet", "densenet"]:
                return "cool"
            else:
                return "mixed"
        
        return "default"

# 创建增强服务实例
enhanced_visualization_service = EnhancedVisualizationService()
