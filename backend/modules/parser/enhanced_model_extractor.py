"""
增强型模型结构提取器
整合所有分析器，提供完整的模型分析功能
"""

import ast
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import re
import importlib.util
import sys
import tempfile
import os

from .ast_analyzer import ASTAnalyzer, ModelInfo, LayerInfo
from .pytorch_inspector import PyTorchInspector, ModelStructure, ModuleInfo
from .dynamic_analyzer import DynamicGraphAnalyzer, DynamicGraphInfo
from .architecture_patterns import ArchitecturePatternAnalyzer, ArchitecturePattern
from .tensor_flow_analyzer import TensorFlowAnalyzer, TensorFlowAnalysis

@dataclass
class ConnectionInfo:
    """连接信息数据类"""
    source: str
    target: str
    connection_type: str  # 'sequential', 'residual', 'attention', 'branch', 'merge'
    data_flow: Dict[str, Any]

@dataclass
class NetworkGraph:
    """网络图数据类"""
    nodes: Dict[str, Dict[str, Any]]
    edges: List[ConnectionInfo]
    input_nodes: List[str]
    output_nodes: List[str]
    graph_type: str  # 'feedforward', 'residual', 'attention', 'complex'

@dataclass
class EnhancedModelInfo:
    """增强模型信息数据类"""
    model_name: str
    ast_info: Optional[ModelInfo]
    structure_info: Optional[ModelStructure] 
    network_graph: NetworkGraph
    complexity_analysis: Dict[str, Any]
    forward_flow_analysis: Dict[str, Any]
    dynamic_analysis: Optional[DynamicGraphInfo]
    architecture_patterns: Optional[ArchitecturePattern]
    tensor_flow_analysis: Optional[TensorFlowAnalysis]

class EnhancedModelExtractor:
    """增强型模型结构提取器"""
    
    def __init__(self):
        self.ast_analyzer = ASTAnalyzer()
        self.pytorch_inspector = PyTorchInspector()
        self.dynamic_analyzer = DynamicGraphAnalyzer()
        self.pattern_analyzer = ArchitecturePatternAnalyzer()
        self.tensor_analyzer = TensorFlowAnalyzer()
        
        # 定义复杂模型模式
        self.complex_patterns = {
            'residual': [
                r'(\w+)\s*\+\s*(\w+)',  # x + residual
                r'torch\.add\((\w+),\s*(\w+)\)',  # torch.add(x, residual)
                r'(\w+)\s*=\s*(\w+)\s*\+\s*self\.(\w+)',  # x = input + self.layer
            ],
            'attention': [
                r'torch\.nn\.MultiheadAttention',
                r'attention\s*\(',
                r'self_attention',
                r'cross_attention',
                r'\.attn\(',
            ],
            'branch': [
                r'if\s+.*:',
                r'torch\.cat\(',
                r'torch\.stack\(',
                r'chunk\(',
                r'split\(',
            ],
            'merge': [
                r'torch\.cat\(',
                r'torch\.stack\(',
                r'torch\.add\(',
                r'torch\.mul\(',
            ]
        }
    
    def extract_from_code(self, code: str, input_shape: Tuple[int, ...] = None) -> EnhancedModelInfo:
        """从代码字符串提取模型信息"""
        if input_shape is None:
            input_shape = (1, 3, 224, 224)
            
        print("开始增强型模型分析...")
        
        # AST分析
        ast_info = None
        try:
            ast_info = self.ast_analyzer.analyze_code(code)
            print("✓ AST分析完成")
        except Exception as e:
            print(f"✗ AST分析失败: {e}")
        
        # 尝试创建模型实例进行内省
        structure_info = None
        model = None
        try:
            model = self._create_model_from_code(code)
            if model:
                structure_info = self.pytorch_inspector.inspect_model(model, input_shape)
                print("✓ PyTorch内省完成")
        except Exception as e:
            print(f"✗ 模型内省失败: {e}")
        
        # 动态图分析
        dynamic_analysis = None
        if model:
            try:
                dynamic_analysis = self.dynamic_analyzer.analyze_dynamic_model(
                    model, input_shape, code
                )
                print("✓ 动态图分析完成")
            except Exception as e:
                print(f"✗ 动态图分析失败: {e}")
        
        # 架构模式分析
        architecture_patterns = None
        if model:
            try:
                execution_order = structure_info.execution_order if structure_info else None
                architecture_patterns = self.pattern_analyzer.analyze_architecture_patterns(
                    model, code, execution_order
                )
                print("✓ 架构模式分析完成")
            except Exception as e:
                print(f"✗ 架构模式分析失败: {e}")
        
        # Tensor流分析
        tensor_flow_analysis = None
        if model:
            try:
                tensor_flow_analysis = self.tensor_analyzer.analyze_tensor_flow(
                    model, code, input_shape
                )
                print("✓ Tensor流分析完成")
            except Exception as e:
                print(f"✗ Tensor流分析失败: {e}")
        
        return self._build_enhanced_info(
            ast_info, structure_info, code, dynamic_analysis, 
            architecture_patterns, tensor_flow_analysis
        )
    
    def extract_from_file(self, file_path: str, input_shape: Tuple[int, ...] = None) -> EnhancedModelInfo:
        """从文件提取模型信息"""
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        return self.extract_from_code(code, input_shape)
    
    def extract_from_model(self, model: nn.Module, input_shape: Tuple[int, ...] = None) -> EnhancedModelInfo:
        """从模型实例提取信息"""
        if input_shape is None:
            input_shape = (1, 3, 224, 224)
            
        print("开始增强型模型分析...")
        
        # 获取模型源代码
        code = ""
        ast_info = None
        try:
            import inspect
            code = inspect.getsource(model.__class__)
            ast_info = self.ast_analyzer.analyze_code(code)
            print("✓ AST分析完成")
        except Exception as e:
            print(f"✗ 获取模型源代码失败: {e}")
        
        # PyTorch内省
        structure_info = self.pytorch_inspector.inspect_model(model, input_shape)
        print("✓ PyTorch内省完成")
        
        # 动态图分析
        dynamic_analysis = None
        try:
            dynamic_analysis = self.dynamic_analyzer.analyze_dynamic_model(
                model, input_shape, code
            )
            print("✓ 动态图分析完成")
        except Exception as e:
            print(f"✗ 动态图分析失败: {e}")
        
        # 架构模式分析
        architecture_patterns = None
        try:
            execution_order = structure_info.execution_order if structure_info else None
            architecture_patterns = self.pattern_analyzer.analyze_architecture_patterns(
                model, code, execution_order
            )
            print("✓ 架构模式分析完成")
        except Exception as e:
            print(f"✗ 架构模式分析失败: {e}")
        
        # Tensor流分析
        tensor_flow_analysis = None
        try:
            tensor_flow_analysis = self.tensor_analyzer.analyze_tensor_flow(
                model, code, input_shape
            )
            print("✓ Tensor流分析完成")
        except Exception as e:
            print(f"✗ Tensor流分析失败: {e}")
        
        return self._build_enhanced_info(
            ast_info, structure_info, code, dynamic_analysis, 
            architecture_patterns, tensor_flow_analysis
        )
    
    def _create_model_from_code(self, code: str) -> Optional[nn.Module]:
        """从代码字符串创建模型实例"""
        try:
            # 创建临时模块
            namespace = {}
            
            # 添加必要的导入
            exec("import torch", namespace)
            exec("import torch.nn as nn", namespace)
            exec("import torch.nn.functional as F", namespace)
            
            # 执行代码
            exec(code, namespace)
            
            # 查找模型类
            model_class = None
            for name, obj in namespace.items():
                if (isinstance(obj, type) and 
                    issubclass(obj, nn.Module) and 
                    obj != nn.Module):
                    model_class = obj
                    break
            
            if model_class:
                # 尝试创建实例
                try:
                    return model_class()
                except:
                    # 尝试带参数创建
                    return model_class(10)  # 假设需要num_classes参数
            
        except Exception as e:
            print(f"创建模型实例失败: {e}")
        
        return None
    
    def _build_enhanced_info(self, 
                           ast_info: Optional[ModelInfo], 
                           structure_info: Optional[ModelStructure], 
                           code: str,
                           dynamic_analysis: Optional[DynamicGraphInfo],
                           architecture_patterns: Optional[ArchitecturePattern],
                           tensor_flow_analysis: Optional[TensorFlowAnalysis]) -> EnhancedModelInfo:
        """构建增强模型信息"""
        
        # 确定模型名称
        model_name = "Unknown"
        if ast_info:
            model_name = ast_info.class_name
        elif structure_info:
            model_name = structure_info.model_name
        
        # 构建增强网络图
        network_graph = self._build_enhanced_network_graph(
            ast_info, structure_info, code, architecture_patterns
        )
        
        # 增强复杂度分析
        complexity_analysis = self._analyze_enhanced_complexity(
            ast_info, structure_info, code, dynamic_analysis, architecture_patterns
        )
        
        # 增强前向流分析
        forward_flow_analysis = self._analyze_enhanced_forward_flow(
            ast_info, structure_info, code, tensor_flow_analysis
        )
        
        return EnhancedModelInfo(
            model_name=model_name,
            ast_info=ast_info,
            structure_info=structure_info,
            network_graph=network_graph,
            complexity_analysis=complexity_analysis,
            forward_flow_analysis=forward_flow_analysis,
            dynamic_analysis=dynamic_analysis,
            architecture_patterns=architecture_patterns,
            tensor_flow_analysis=tensor_flow_analysis
        )
    
    def _build_enhanced_network_graph(self, 
                                    ast_info: Optional[ModelInfo], 
                                    structure_info: Optional[ModelStructure], 
                                    code: str,
                                    architecture_patterns: Optional[ArchitecturePattern]) -> NetworkGraph:
        """构建增强网络图"""
        nodes = {}
        edges = []
        
        # 从结构信息构建节点
        if structure_info:
            for name, module_info in structure_info.modules.items():
                nodes[name] = {
                    'id': name,
                    'type': module_info.module_type,
                    'label': f"{module_info.module_type}",
                    'parameters': module_info.parameters,
                    'input_shape': module_info.input_shape,
                    'output_shape': module_info.output_shape,
                    'num_parameters': module_info.num_parameters,
                    'trainable': module_info.trainable_parameters > 0
                }
        
        # 从AST信息补充节点
        if ast_info:
            for layer in ast_info.layers:
                if layer.name not in nodes:
                    nodes[layer.name] = {
                        'id': layer.name,
                        'type': layer.layer_type,
                        'label': f"{layer.layer_type}({', '.join(map(str, layer.args))})",
                        'parameters': layer.kwargs,
                        'input_shape': None,
                        'output_shape': None,
                        'num_parameters': 0,
                        'trainable': True
                    }
        
        # 提取基础连接
        edges = self._extract_enhanced_connections(
            ast_info, structure_info, code, architecture_patterns
        )
        
        # 识别输入输出节点
        input_nodes, output_nodes = self._identify_io_nodes(nodes, edges, structure_info)
        
        # 确定图类型
        graph_type = self._determine_enhanced_graph_type(edges, code, architecture_patterns)
        
        return NetworkGraph(
            nodes=nodes,
            edges=edges,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            graph_type=graph_type
        )
    
    def _extract_enhanced_connections(self, 
                                    ast_info: Optional[ModelInfo], 
                                    structure_info: Optional[ModelStructure], 
                                    code: str,
                                    architecture_patterns: Optional[ArchitecturePattern]) -> List[ConnectionInfo]:
        """提取增强连接关系"""
        connections = []
        
        # 从执行顺序推断连接
        if structure_info and structure_info.execution_order:
            for i in range(len(structure_info.execution_order) - 1):
                source = structure_info.execution_order[i]
                target = structure_info.execution_order[i + 1]
                
                connections.append(ConnectionInfo(
                    source=source,
                    target=target,
                    connection_type='sequential',
                    data_flow={'type': 'forward'}
                ))
        
        # 从架构模式添加连接
        if architecture_patterns:
            # 添加残差连接
            for conn in architecture_patterns.residual_connections:
                connections.append(ConnectionInfo(
                    source=conn.input_layer,
                    target=conn.output_layer,
                    connection_type='residual',
                    data_flow={
                        'operation': conn.operation,
                        'shortcut': conn.shortcut_layer
                    }
                ))
            
            # 添加密集连接
            for conn in architecture_patterns.dense_connections:
                for source in conn.source_layers:
                    connections.append(ConnectionInfo(
                        source=source,
                        target=conn.target_layer,
                        connection_type='dense',
                        data_flow={'operation': conn.operation}
                    ))
            
            # 添加注意力连接
            for attn in architecture_patterns.attention_patterns:
                connections.extend([
                    ConnectionInfo(
                        source=attn.query_layer,
                        target=attn.output_layer,
                        connection_type='attention_query',
                        data_flow={'type': 'query'}
                    ),
                    ConnectionInfo(
                        source=attn.key_layer,
                        target=attn.output_layer,
                        connection_type='attention_key',
                        data_flow={'type': 'key'}
                    ),
                    ConnectionInfo(
                        source=attn.value_layer,
                        target=attn.output_layer,
                        connection_type='attention_value',
                        data_flow={'type': 'value'}
                    )
                ])
        
        return connections
    
    def _identify_io_nodes(self, nodes: Dict[str, Dict], 
                          edges: List[ConnectionInfo], 
                          structure_info: Optional[ModelStructure]) -> Tuple[List[str], List[str]]:
        """识别输入输出节点"""
        input_nodes = []
        output_nodes = []
        
        if structure_info and structure_info.execution_order:
            if structure_info.execution_order:
                input_nodes.append(structure_info.execution_order[0])
                output_nodes.append(structure_info.execution_order[-1])
        
        return input_nodes, output_nodes
    
    def _determine_enhanced_graph_type(self, edges: List[ConnectionInfo], 
                                     code: str,
                                     architecture_patterns: Optional[ArchitecturePattern]) -> str:
        """确定增强图类型"""
        if architecture_patterns:
            return architecture_patterns.pattern_type
        
        has_residual = any(edge.connection_type == 'residual' for edge in edges)
        has_attention = any('attention' in edge.connection_type for edge in edges)
        has_branch = any(edge.connection_type == 'branch' for edge in edges)
        
        if has_attention:
            return 'transformer'
        elif has_residual:
            return 'resnet'
        elif has_branch:
            return 'complex'
        else:
            return 'feedforward'
    
    def _analyze_enhanced_complexity(self, 
                                   ast_info: Optional[ModelInfo], 
                                   structure_info: Optional[ModelStructure], 
                                   code: str,
                                   dynamic_analysis: Optional[DynamicGraphInfo],
                                   architecture_patterns: Optional[ArchitecturePattern]) -> Dict[str, Any]:
        """分析增强复杂度"""
        complexity = {
            'total_layers': 0,
            'layer_types': {},
            'has_residual': False,
            'has_attention': False,
            'has_branching': False,
            'has_dynamic_control': False,
            'depth': 0,
            'width': 0
        }
        
        if structure_info:
            complexity.update({
                'total_parameters': structure_info.total_parameters,
                'trainable_parameters': structure_info.trainable_parameters,
                'model_size_mb': structure_info.model_size_mb,
                'total_layers': len(structure_info.modules)
            })
            
            # 统计层类型
            for module_info in structure_info.modules.values():
                layer_type = module_info.module_type
                complexity['layer_types'][layer_type] = complexity['layer_types'].get(layer_type, 0) + 1
        
        # 从架构模式更新复杂度
        if architecture_patterns:
            complexity['has_residual'] = len(architecture_patterns.residual_connections) > 0
            complexity['has_attention'] = len(architecture_patterns.attention_patterns) > 0
            complexity['has_branching'] = len(architecture_patterns.branch_points) > 0
            complexity['architecture_type'] = architecture_patterns.pattern_type
        
        # 从动态分析更新复杂度
        if dynamic_analysis:
            complexity['has_dynamic_control'] = dynamic_analysis.has_dynamic_control
            complexity['control_flow_nodes'] = len(dynamic_analysis.control_flow_nodes)
            complexity['execution_paths'] = len(dynamic_analysis.execution_paths)
        
        return complexity
    
    def _analyze_enhanced_forward_flow(self, 
                                     ast_info: Optional[ModelInfo], 
                                     structure_info: Optional[ModelStructure], 
                                     code: str,
                                     tensor_flow_analysis: Optional[TensorFlowAnalysis]) -> Dict[str, Any]:
        """分析增强前向传播流"""
        flow_analysis = {
            'execution_order': [],
            'data_transformations': [],
            'branching_points': [],
            'merge_points': [],
            'activation_functions': [],
            'tensor_operations': [],
            'shape_changes': {}
        }
        
        if structure_info:
            flow_analysis['execution_order'] = structure_info.execution_order
        
        if ast_info:
            flow_analysis['data_transformations'] = ast_info.forward_flow
            
            # 识别激活函数
            for flow in ast_info.forward_flow:
                for activation in ['relu', 'sigmoid', 'tanh', 'softmax', 'gelu']:
                    if activation in flow.lower():
                        flow_analysis['activation_functions'].append(activation)
        
        # 从tensor流分析更新
        if tensor_flow_analysis:
            flow_analysis['tensor_operations'] = [
                {
                    'id': op.op_id,
                    'type': op.op_type,
                    'source': op.source_layer
                } for op in tensor_flow_analysis.tensor_operations
            ]
            flow_analysis['shape_changes'] = tensor_flow_analysis.shape_changes
            flow_analysis['data_dependencies'] = tensor_flow_analysis.data_dependencies
        
        return flow_analysis
    
    def get_enhanced_visualization_data(self, model_info: EnhancedModelInfo) -> Dict[str, Any]:
        """获取增强可视化数据"""
        viz_data = {
            'nodes': list(model_info.network_graph.nodes.values()),
            'edges': [asdict(edge) for edge in model_info.network_graph.edges],
            'metadata': {
                'model_name': model_info.model_name,
                'graph_type': model_info.network_graph.graph_type,
                'complexity': model_info.complexity_analysis,
                'input_nodes': model_info.network_graph.input_nodes,
                'output_nodes': model_info.network_graph.output_nodes
            }
        }
        
        # 添加动态分析数据
        if model_info.dynamic_analysis:
            viz_data['dynamic_info'] = self.dynamic_analyzer.get_dynamic_visualization_data(
                model_info.dynamic_analysis
            )
        
        # 添加架构模式数据
        if model_info.architecture_patterns:
            viz_data['architecture_patterns'] = self.pattern_analyzer.get_architecture_visualization_data(
                model_info.architecture_patterns
            )
        
        # 添加tensor流数据
        if model_info.tensor_flow_analysis:
            viz_data['tensor_flow'] = self.tensor_analyzer.get_tensor_flow_visualization_data(
                model_info.tensor_flow_analysis
            )
        
        return viz_data
    
    def to_dict(self, model_info: EnhancedModelInfo) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(model_info)
