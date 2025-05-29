"""
连接增强器
专门处理PyTorch模型中的隐式连接，如view操作
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ConnectionInfo:
    """连接信息数据类"""
    source: str
    target: str
    connection_type: str
    data_flow: Dict[str, Any]

class ConnectionEnhancer:
    """连接增强器，专门处理隐式连接"""
    
    def __init__(self):
        # 形状变换操作的模式
        self.reshape_patterns = [
            r'x\s*=\s*x\.view\s*\(',
            r'x\s*=\s*x\.reshape\s*\(',
            r'x\s*=\s*torch\.flatten\s*\(',
            r'x\s*=\s*x\.flatten\s*\(',
            r'x\s*=\s*x\.view\s*\(\s*x\.size\s*\(\s*0\s*\)\s*,\s*-1\s*\)',
        ]
        
        # 常见的激活函数模式
        self.activation_patterns = [
            r'x\s*=\s*self\.relu\s*\(\s*(\w+)\s*\)',
            r'x\s*=\s*F\.relu\s*\(\s*(\w+)\s*\)',
            r'x\s*=\s*torch\.relu\s*\(\s*(\w+)\s*\)',
        ]
        
        # 池化操作模式
        self.pooling_patterns = [
            r'x\s*=\s*self\.pool\s*\(\s*(\w+)\s*\)',
            r'x\s*=\s*F\.max_pool2d\s*\(\s*(\w+)\s*',
            r'x\s*=\s*F\.avg_pool2d\s*\(\s*(\w+)\s*',
        ]
    
    def enhance_connections(self, code: str, existing_connections: List[ConnectionInfo], 
                          execution_order: List[str]) -> List[ConnectionInfo]:
        """增强连接关系"""
        enhanced_connections = existing_connections.copy()
        
        # 检测形状变换连接
        enhanced_connections.extend(self._detect_reshape_connections(code, execution_order))
        
        # 检测激活函数连接
        enhanced_connections.extend(self._detect_activation_connections(code, execution_order))
        
        # 检测池化连接
        enhanced_connections.extend(self._detect_pooling_connections(code, execution_order))
        
        # 去重
        enhanced_connections = self._remove_duplicates(enhanced_connections)
        
        return enhanced_connections
    
    def _detect_reshape_connections(self, code: str, execution_order: List[str]) -> List[ConnectionInfo]:
        """检测形状变换连接"""
        connections = []
        
        # 检查是否有view操作
        has_view_operation = any(re.search(pattern, code) for pattern in self.reshape_patterns)
        
        if has_view_operation and execution_order:
            # 找到最后的卷积相关层和第一个线性层
            conv_related = []
            linear_layers = []
            
            for layer_name in execution_order:
                layer_lower = layer_name.lower()
                if any(keyword in layer_lower for keyword in ['conv', 'pool', 'norm']):
                    conv_related.append(layer_name)
                elif any(keyword in layer_lower for keyword in ['fc', 'linear', 'classifier']):
                    linear_layers.append(layer_name)
            
            if conv_related and linear_layers:
                last_conv = conv_related[-1]
                first_linear = linear_layers[0]
                
                connections.append(ConnectionInfo(
                    source=last_conv,
                    target=first_linear,
                    connection_type='reshape',
                    data_flow={'operation': 'view', 'type': 'flatten'}
                ))
        
        return connections
    
    def _detect_activation_connections(self, code: str, execution_order: List[str]) -> List[ConnectionInfo]:
        """检测激活函数连接"""
        connections = []
        
        # 这里可以根据需要添加激活函数的连接逻辑
        # 目前简化处理
        
        return connections
    
    def _detect_pooling_connections(self, code: str, execution_order: List[str]) -> List[ConnectionInfo]:
        """检测池化连接"""
        connections = []
        
        # 这里可以根据需要添加池化层的连接逻辑
        # 目前简化处理
        
        return connections
    
    def _remove_duplicates(self, connections: List[ConnectionInfo]) -> List[ConnectionInfo]:
        """去除重复连接"""
        seen = set()
        unique_connections = []
        
        for conn in connections:
            key = (conn.source, conn.target, conn.connection_type)
            if key not in seen:
                seen.add(key)
                unique_connections.append(conn)
        
        return unique_connections
    
    def analyze_forward_flow(self, code: str) -> Dict[str, Any]:
        """分析前向传播流程"""
        analysis = {
            'has_reshape': False,
            'reshape_locations': [],
            'activation_count': 0,
            'pooling_count': 0,
            'complex_operations': []
        }
        
        # 检测形状变换
        for pattern in self.reshape_patterns:
            matches = re.findall(pattern, code)
            if matches:
                analysis['has_reshape'] = True
                analysis['reshape_locations'].extend(matches)
        
        # 检测激活函数
        for pattern in self.activation_patterns:
            matches = re.findall(pattern, code)
            analysis['activation_count'] += len(matches)
        
        # 检测池化操作
        for pattern in self.pooling_patterns:
            matches = re.findall(pattern, code)
            analysis['pooling_count'] += len(matches)
        
        return analysis
