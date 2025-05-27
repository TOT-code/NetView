"""
模型结构提取器
整合AST分析和PyTorch内省，提取完整的模型结构信息
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
class CompleteModelInfo:
    """完整模型信息数据类"""
    model_name: str
    ast_info: Optional[ModelInfo]
    structure_info: Optional[ModelStructure] 
    network_graph: NetworkGraph
    complexity_analysis: Dict[str, Any]
    forward_flow_analysis: Dict[str, Any]
    dynamic_analysis: Optional[DynamicGraphInfo]
    architecture_patterns: Optional[ArchitecturePattern]
    tensor_flow_analysis: Optional[TensorFlowAnalysis]

class ModelExtractor:
    """模型结构提取器"""
    
    def __init__(self):
        self.ast_analyzer = ASTAnalyzer()
        self.pytorch_inspector = PyTorchInspector()
        
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
    
    def extract_from_code(self, code: str, input_shape: Tuple[int, ...] = None) -> CompleteModelInfo:
        """从代码字符串提取模型信息"""
        # AST分析
        try:
            ast_info = self.ast_analyzer.analyze_code(code)
        except Exception as e:
            print(f"AST分析失败: {e}")
            ast_info = None
        
        # 尝试创建模型实例进行内省
        structure_info = None
        try:
            model = self._create_model_from_code(code)
            if model:
                structure_info = self.pytorch_inspector.inspect_model(model, input_shape)
        except Exception as e:
            print(f"模型内省失败: {e}")
        
        return self._build_complete_info(ast_info, structure_info, code)
    
    def extract_from_file(self, file_path: str, input_shape: Tuple[int, ...] = None) -> CompleteModelInfo:
        """从文件提取模型信息"""
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # AST分析
        try:
            ast_info = self.ast_analyzer.analyze_file(file_path)
        except Exception as e:
            print(f"AST分析失败: {e}")
            ast_info = None
        
        # 尝试导入模型进行内省
        structure_info = None
        try:
            model = self._import_model_from_file(file_path)
            if model:
                structure_info = self.pytorch_inspector.inspect_model(model, input_shape)
        except Exception as e:
            print(f"模型内省失败: {e}")
        
        return self._build_complete_info(ast_info, structure_info, code)
    
    def extract_from_model(self, model: nn.Module, input_shape: Tuple[int, ...] = None) -> CompleteModelInfo:
        """从模型实例提取信息"""
        # 获取模型源代码
        try:
            import inspect
            code = inspect.getsource(model.__class__)
            ast_info = self.ast_analyzer.analyze_code(code)
        except Exception as e:
            print(f"获取模型源代码失败: {e}")
            ast_info = None
            code = ""
        
        # PyTorch内省
        structure_info = self.pytorch_inspector.inspect_model(model, input_shape)
        
        return self._build_complete_info(ast_info, structure_info, code)
    
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
    
    def _import_model_from_file(self, file_path: str) -> Optional[nn.Module]:
        """从文件导入模型"""
        try:
            spec = importlib.util.spec_from_file_location("model_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找模型类
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, nn.Module) and 
                    obj != nn.Module):
                    try:
                        return obj()
                    except:
                        try:
                            return obj(10)
                        except:
                            continue
        except Exception as e:
            print(f"导入模型失败: {e}")
        
        return None
    
    def _build_complete_info(self, ast_info: Optional[ModelInfo], 
                           structure_info: Optional[ModelStructure], 
                           code: str) -> CompleteModelInfo:
        """构建完整模型信息"""
        
        # 确定模型名称
        model_name = "Unknown"
        if ast_info:
            model_name = ast_info.class_name
        elif structure_info:
            model_name = structure_info.model_name
        
        # 构建网络图
        network_graph = self._build_network_graph(ast_info, structure_info, code)
        
        # 复杂度分析
        complexity_analysis = self._analyze_complexity(ast_info, structure_info, code)
        
        # 前向流分析
        forward_flow_analysis = self._analyze_forward_flow(ast_info, structure_info, code)
        
        return CompleteModelInfo(
            model_name=model_name,
            ast_info=ast_info,
            structure_info=structure_info,
            network_graph=network_graph,
            complexity_analysis=complexity_analysis,
            forward_flow_analysis=forward_flow_analysis
        )
    
    def _build_network_graph(self, ast_info: Optional[ModelInfo], 
                           structure_info: Optional[ModelStructure], 
                           code: str) -> NetworkGraph:
        """构建网络图"""
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
        
        # 分析连接关系
        edges = self._extract_connections(ast_info, structure_info, code)
        
        # 识别输入输出节点
        input_nodes, output_nodes = self._identify_io_nodes(nodes, edges, structure_info)
        
        # 确定图类型
        graph_type = self._determine_graph_type(edges, code)
        
        return NetworkGraph(
            nodes=nodes,
            edges=edges,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            graph_type=graph_type
        )
    
    def _extract_connections(self, ast_info: Optional[ModelInfo], 
                           structure_info: Optional[ModelStructure], 
                           code: str) -> List[ConnectionInfo]:
        """提取连接关系"""
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
        
        # 从AST分析推断复杂连接
        if ast_info and ast_info.forward_flow:
            connections.extend(self._analyze_forward_connections(ast_info.forward_flow))
        
        # 从代码分析特殊连接模式
        connections.extend(self._detect_complex_connections(code))
        
        return connections
    
    def _analyze_forward_connections(self, forward_flow: List[str]) -> List[ConnectionInfo]:
        """分析前向传播连接"""
        connections = []
        
        for flow in forward_flow:
            # 检测残差连接
            for pattern in self.complex_patterns['residual']:
                matches = re.findall(pattern, flow)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        connections.append(ConnectionInfo(
                            source=match[0],
                            target=match[1] if len(match) > 2 else 'output',
                            connection_type='residual',
                            data_flow={'operation': 'add'}
                        ))
            
            # 检测分支连接
            for pattern in self.complex_patterns['branch']:
                if re.search(pattern, flow):
                    connections.append(ConnectionInfo(
                        source='input',
                        target='branch',
                        connection_type='branch',
                        data_flow={'operation': 'split'}
                    ))
        
        return connections
    
    def _detect_complex_connections(self, code: str) -> List[ConnectionInfo]:
        """检测复杂连接模式"""
        connections = []
        
        # 检测注意力机制
        for pattern in self.complex_patterns['attention']:
            if re.search(pattern, code):
                connections.append(ConnectionInfo(
                    source='query',
                    target='attention_output',
                    connection_type='attention',
                    data_flow={'mechanism': 'self_attention'}
                ))
        
        return connections
    
    def _identify_io_nodes(self, nodes: Dict[str, Dict], 
                          edges: List[ConnectionInfo], 
                          structure_info: Optional[ModelStructure]) -> Tuple[List[str], List[str]]:
        """识别输入输出节点"""
        input_nodes = []
        output_nodes = []
        
        if structure_info and structure_info.execution_order:
            # 第一个执行的模块通常是输入相关
            if structure_info.execution_order:
                input_nodes.append(structure_info.execution_order[0])
                output_nodes.append(structure_info.execution_order[-1])
        
        return input_nodes, output_nodes
    
    def _determine_graph_type(self, edges: List[ConnectionInfo], code: str) -> str:
        """确定图类型"""
        has_residual = any(edge.connection_type == 'residual' for edge in edges)
        has_attention = any(edge.connection_type == 'attention' for edge in edges)
        has_branch = any(edge.connection_type == 'branch' for edge in edges)
        
        if has_attention:
            return 'attention'
        elif has_residual:
            return 'residual'
        elif has_branch:
            return 'complex'
        else:
            return 'feedforward'
    
    def _analyze_complexity(self, ast_info: Optional[ModelInfo], 
                          structure_info: Optional[ModelStructure], 
                          code: str) -> Dict[str, Any]:
        """分析模型复杂度"""
        complexity = {
            'total_layers': 0,
            'layer_types': {},
            'has_residual': False,
            'has_attention': False,
            'has_branching': False,
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
        
        # 检测复杂结构
        complexity['has_residual'] = any(
            re.search(pattern, code) for pattern in self.complex_patterns['residual']
        )
        complexity['has_attention'] = any(
            re.search(pattern, code) for pattern in self.complex_patterns['attention']
        )
        complexity['has_branching'] = any(
            re.search(pattern, code) for pattern in self.complex_patterns['branch']
        )
        
        return complexity
    
    def _analyze_forward_flow(self, ast_info: Optional[ModelInfo], 
                            structure_info: Optional[ModelStructure], 
                            code: str) -> Dict[str, Any]:
        """分析前向传播流"""
        flow_analysis = {
            'execution_order': [],
            'data_transformations': [],
            'branching_points': [],
            'merge_points': [],
            'activation_functions': []
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
        
        return flow_analysis
    
    def to_dict(self, model_info: CompleteModelInfo) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(model_info)
    
    def get_visualization_data(self, model_info: CompleteModelInfo) -> Dict[str, Any]:
        """获取可视化数据"""
        return {
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
