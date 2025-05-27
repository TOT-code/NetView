"""
架构模式分析器
识别和处理ResNet、DenseNet、Transformer等复杂架构模式
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import re
import ast

@dataclass
class ResidualConnection:
    """残差连接"""
    connection_id: str
    input_layer: str
    output_layer: str
    shortcut_layer: Optional[str]
    operation: str  # 'add', 'concat', 'multiply'
    
@dataclass
class DenseConnection:
    """密集连接"""
    connection_id: str
    source_layers: List[str]
    target_layer: str
    operation: str  # 'concat', 'sum'

@dataclass
class AttentionPattern:
    """注意力模式"""
    pattern_id: str
    attention_type: str  # 'self', 'cross', 'multi_head'
    query_layer: str
    key_layer: str
    value_layer: str
    output_layer: str

@dataclass
class ArchitecturePattern:
    """架构模式"""
    pattern_type: str  # 'resnet', 'densenet', 'transformer', 'unet', 'inception'
    residual_connections: List[ResidualConnection]
    dense_connections: List[DenseConnection]
    attention_patterns: List[AttentionPattern]
    branch_points: List[Dict[str, Any]]
    merge_points: List[Dict[str, Any]]

class ArchitecturePatternAnalyzer:
    """架构模式分析器"""
    
    def __init__(self):
        # ResNet模式
        self.resnet_patterns = [
            r'(\w+)\s*\+\s*(\w+)',  # x + shortcut
            r'torch\.add\((\w+),\s*(\w+)\)',
            r'(\w+)\s*=\s*(\w+)\s*\+\s*self\.shortcut\(',
            r'out\s*\+=\s*(\w+)',
            r'identity\s*=',
            r'shortcut\s*=',
        ]
        
        # DenseNet模式
        self.densenet_patterns = [
            r'torch\.cat\(',
            r'torch\.concat\(',
            r'cat\(.*dim\s*=\s*1',
            r'concat_features',
            r'dense_layer',
        ]
        
        # Attention模式
        self.attention_patterns = [
            r'MultiheadAttention',
            r'self_attention',
            r'cross_attention',
            r'\.attn\(',
            r'attention_weights',
            r'scaled_dot_product',
            r'query.*key.*value',
        ]
        
        # 分支模式
        self.branch_patterns = [
            r'torch\.chunk\(',
            r'torch\.split\(',
            r'\.chunk\(',
            r'\.split\(',
            r'if.*else',
        ]
        
        # 合并模式
        self.merge_patterns = [
            r'torch\.cat\(',
            r'torch\.stack\(',
            r'torch\.add\(',
            r'torch\.mul\(',
            r'\+',
            r'\*',
        ]
    
    def analyze_architecture_patterns(self, model: nn.Module, 
                                    code: str = None,
                                    execution_order: List[str] = None) -> ArchitecturePattern:
        """分析架构模式"""
        
        # 检测残差连接
        residual_connections = self._detect_residual_connections(model, code, execution_order)
        
        # 检测密集连接
        dense_connections = self._detect_dense_connections(model, code, execution_order)
        
        # 检测注意力模式
        attention_patterns = self._detect_attention_patterns(model, code)
        
        # 检测分支点
        branch_points = self._detect_branch_points(model, code)
        
        # 检测合并点
        merge_points = self._detect_merge_points(model, code)
        
        # 确定架构类型
        pattern_type = self._determine_architecture_type(
            residual_connections, dense_connections, attention_patterns
        )
        
        return ArchitecturePattern(
            pattern_type=pattern_type,
            residual_connections=residual_connections,
            dense_connections=dense_connections,
            attention_patterns=attention_patterns,
            branch_points=branch_points,
            merge_points=merge_points
        )
    
    def _detect_residual_connections(self, model: nn.Module, 
                                   code: str = None,
                                   execution_order: List[str] = None) -> List[ResidualConnection]:
        """检测残差连接"""
        connections = []
        
        # 从代码分析
        if code:
            connections.extend(self._analyze_residual_from_code(code))
        
        # 从模型结构分析
        connections.extend(self._analyze_residual_from_model(model))
        
        # 从执行顺序分析
        if execution_order:
            connections.extend(self._analyze_residual_from_execution(execution_order))
        
        return connections
    
    def _analyze_residual_from_code(self, code: str) -> List[ResidualConnection]:
        """从代码分析残差连接"""
        connections = []
        connection_id = 0
        
        for pattern in self.resnet_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                connection_id += 1
                
                if len(match.groups()) >= 2:
                    connections.append(ResidualConnection(
                        connection_id=f'residual_{connection_id}',
                        input_layer=match.group(1),
                        output_layer=match.group(2) if len(match.groups()) > 1 else 'output',
                        shortcut_layer=None,
                        operation='add'
                    ))
        
        return connections
    
    def _analyze_residual_from_model(self, model: nn.Module) -> List[ResidualConnection]:
        """从模型结构分析残差连接"""
        connections = []
        connection_id = 0
        
        # 查找ResNet块模式
        for name, module in model.named_modules():
            if self._is_residual_block(module):
                connection_id += 1
                
                # 检查是否有shortcut层
                shortcut_layer = None
                if hasattr(module, 'shortcut'):
                    shortcut_layer = f"{name}.shortcut"
                elif hasattr(module, 'downsample'):
                    shortcut_layer = f"{name}.downsample"
                
                connections.append(ResidualConnection(
                    connection_id=f'model_residual_{connection_id}',
                    input_layer=name,
                    output_layer=name,
                    shortcut_layer=shortcut_layer,
                    operation='add'
                ))
        
        return connections
    
    def _is_residual_block(self, module: nn.Module) -> bool:
        """判断是否为残差块"""
        module_name = module.__class__.__name__.lower()
        
        # 常见的残差块名称
        residual_keywords = ['resnet', 'residual', 'basicblock', 'bottleneck', 'resblock']
        
        for keyword in residual_keywords:
            if keyword in module_name:
                return True
        
        # 检查是否有shortcut相关属性
        if hasattr(module, 'shortcut') or hasattr(module, 'downsample'):
            return True
        
        return False
    
    def _analyze_residual_from_execution(self, execution_order: List[str]) -> List[ResidualConnection]:
        """从执行顺序分析残差连接"""
        connections = []
        
        # 寻找可能的跳跃连接模式
        # 如果一个层的输出在后面几层被重新使用，可能是残差连接
        for i, layer1 in enumerate(execution_order):
            for j in range(i + 2, min(i + 10, len(execution_order))):  # 查看后续2-10层
                layer2 = execution_order[j]
                
                # 简单的启发式：如果层名相似，可能有连接
                if self._layers_likely_connected(layer1, layer2):
                    connections.append(ResidualConnection(
                        connection_id=f'exec_residual_{i}_{j}',
                        input_layer=layer1,
                        output_layer=layer2,
                        shortcut_layer=None,
                        operation='add'
                    ))
        
        return connections
    
    def _layers_likely_connected(self, layer1: str, layer2: str) -> bool:
        """判断两层是否可能有残差连接"""
        # 简单的启发式规则
        if 'conv' in layer1.lower() and 'conv' in layer2.lower():
            # 卷积层之间可能有连接
            return True
        
        if layer1.split('.')[0] == layer2.split('.')[0]:
            # 同一个模块内的层
            return True
        
        return False
    
    def _detect_dense_connections(self, model: nn.Module, 
                                code: str = None,
                                execution_order: List[str] = None) -> List[DenseConnection]:
        """检测密集连接"""
        connections = []
        
        # 从代码分析
        if code:
            connections.extend(self._analyze_dense_from_code(code))
        
        # 从模型结构分析
        connections.extend(self._analyze_dense_from_model(model))
        
        return connections
    
    def _analyze_dense_from_code(self, code: str) -> List[DenseConnection]:
        """从代码分析密集连接"""
        connections = []
        connection_id = 0
        
        for pattern in self.densenet_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                connection_id += 1
                
                connections.append(DenseConnection(
                    connection_id=f'dense_{connection_id}',
                    source_layers=['multiple_sources'],  # 需要更详细的分析
                    target_layer='concat_output',
                    operation='concat'
                ))
        
        return connections
    
    def _analyze_dense_from_model(self, model: nn.Module) -> List[DenseConnection]:
        """从模型结构分析密集连接"""
        connections = []
        connection_id = 0
        
        for name, module in model.named_modules():
            if self._is_dense_block(module):
                connection_id += 1
                
                connections.append(DenseConnection(
                    connection_id=f'model_dense_{connection_id}',
                    source_layers=[name],
                    target_layer=name,
                    operation='concat'
                ))
        
        return connections
    
    def _is_dense_block(self, module: nn.Module) -> bool:
        """判断是否为密集块"""
        module_name = module.__class__.__name__.lower()
        dense_keywords = ['dense', 'densenet', 'denseblock', 'densely']
        
        for keyword in dense_keywords:
            if keyword in module_name:
                return True
        
        return False
    
    def _detect_attention_patterns(self, model: nn.Module, code: str = None) -> List[AttentionPattern]:
        """检测注意力模式"""
        patterns = []
        
        # 从代码分析
        if code:
            patterns.extend(self._analyze_attention_from_code(code))
        
        # 从模型结构分析
        patterns.extend(self._analyze_attention_from_model(model))
        
        return patterns
    
    def _analyze_attention_from_code(self, code: str) -> List[AttentionPattern]:
        """从代码分析注意力模式"""
        patterns = []
        pattern_id = 0
        
        for pattern in self.attention_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                pattern_id += 1
                
                patterns.append(AttentionPattern(
                    pattern_id=f'attention_{pattern_id}',
                    attention_type='self',  # 默认自注意力
                    query_layer='query',
                    key_layer='key',
                    value_layer='value',
                    output_layer='attention_output'
                ))
        
        return patterns
    
    def _analyze_attention_from_model(self, model: nn.Module) -> List[AttentionPattern]:
        """从模型结构分析注意力模式"""
        patterns = []
        pattern_id = 0
        
        for name, module in model.named_modules():
            if self._is_attention_module(module):
                pattern_id += 1
                
                attention_type = 'self'
                if isinstance(module, nn.MultiheadAttention):
                    attention_type = 'multi_head'
                
                patterns.append(AttentionPattern(
                    pattern_id=f'model_attention_{pattern_id}',
                    attention_type=attention_type,
                    query_layer=f"{name}.query",
                    key_layer=f"{name}.key",
                    value_layer=f"{name}.value",
                    output_layer=f"{name}.output"
                ))
        
        return patterns
    
    def _is_attention_module(self, module: nn.Module) -> bool:
        """判断是否为注意力模块"""
        if isinstance(module, nn.MultiheadAttention):
            return True
        
        module_name = module.__class__.__name__.lower()
        attention_keywords = ['attention', 'attn', 'transformer']
        
        for keyword in attention_keywords:
            if keyword in module_name:
                return True
        
        return False
    
    def _detect_branch_points(self, model: nn.Module, code: str = None) -> List[Dict[str, Any]]:
        """检测分支点"""
        branch_points = []
        
        if code:
            for pattern in self.branch_patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    branch_points.append({
                        'type': 'branch',
                        'operation': match.group(0),
                        'pattern': pattern,
                        'source': 'code'
                    })
        
        return branch_points
    
    def _detect_merge_points(self, model: nn.Module, code: str = None) -> List[Dict[str, Any]]:
        """检测合并点"""
        merge_points = []
        
        if code:
            for pattern in self.merge_patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    merge_points.append({
                        'type': 'merge',
                        'operation': match.group(0),
                        'pattern': pattern,
                        'source': 'code'
                    })
        
        return merge_points
    
    def _determine_architecture_type(self, residual_connections: List[ResidualConnection],
                                   dense_connections: List[DenseConnection],
                                   attention_patterns: List[AttentionPattern]) -> str:
        """确定架构类型"""
        
        if attention_patterns:
            return 'transformer'
        elif residual_connections and len(residual_connections) > 2:
            return 'resnet'
        elif dense_connections:
            return 'densenet'
        elif residual_connections:
            return 'skip_connection'
        else:
            return 'feedforward'
    
    def get_architecture_visualization_data(self, pattern: ArchitecturePattern) -> Dict[str, Any]:
        """获取架构模式的可视化数据"""
        viz_data = {
            'architecture_type': pattern.pattern_type,
            'residual_connections': [],
            'dense_connections': [],
            'attention_patterns': [],
            'branch_points': pattern.branch_points,
            'merge_points': pattern.merge_points
        }
        
        # 残差连接
        for conn in pattern.residual_connections:
            viz_data['residual_connections'].append({
                'id': conn.connection_id,
                'input': conn.input_layer,
                'output': conn.output_layer,
                'shortcut': conn.shortcut_layer,
                'operation': conn.operation
            })
        
        # 密集连接
        for conn in pattern.dense_connections:
            viz_data['dense_connections'].append({
                'id': conn.connection_id,
                'sources': conn.source_layers,
                'target': conn.target_layer,
                'operation': conn.operation
            })
        
        # 注意力模式
        for attn in pattern.attention_patterns:
            viz_data['attention_patterns'].append({
                'id': attn.pattern_id,
                'type': attn.attention_type,
                'query': attn.query_layer,
                'key': attn.key_layer,
                'value': attn.value_layer,
                'output': attn.output_layer
            })
        
        return viz_data
    
    def enhance_graph_with_patterns(self, base_graph: Dict[str, Any], 
                                   pattern: ArchitecturePattern) -> Dict[str, Any]:
        """用架构模式增强基础图"""
        enhanced_graph = base_graph.copy()
        
        # 添加残差连接边
        for conn in pattern.residual_connections:
            enhanced_graph['edges'].append({
                'source': conn.input_layer,
                'target': conn.output_layer,
                'connection_type': 'residual',
                'data_flow': {
                    'operation': conn.operation,
                    'shortcut': conn.shortcut_layer
                }
            })
        
        # 添加密集连接边
        for conn in pattern.dense_connections:
            for source in conn.source_layers:
                enhanced_graph['edges'].append({
                    'source': source,
                    'target': conn.target_layer,
                    'connection_type': 'dense',
                    'data_flow': {
                        'operation': conn.operation
                    }
                })
        
        # 添加注意力连接
        for attn in pattern.attention_patterns:
            # Q-K-V连接
            enhanced_graph['edges'].extend([
                {
                    'source': attn.query_layer,
                    'target': attn.output_layer,
                    'connection_type': 'attention_query',
                    'data_flow': {'type': 'query'}
                },
                {
                    'source': attn.key_layer,
                    'target': attn.output_layer,
                    'connection_type': 'attention_key',
                    'data_flow': {'type': 'key'}
                },
                {
                    'source': attn.value_layer,
                    'target': attn.output_layer,
                    'connection_type': 'attention_value',
                    'data_flow': {'type': 'value'}
                }
            ])
        
        # 更新元数据
        enhanced_graph['metadata']['architecture_patterns'] = self.get_architecture_visualization_data(pattern)
        
        return enhanced_graph
