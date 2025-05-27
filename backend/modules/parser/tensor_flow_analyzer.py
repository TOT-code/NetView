"""
Tensor流分析器
详细追踪tensor操作，包括reshape、split、concat等数据变换
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import re
import ast

@dataclass
class TensorOperation:
    """Tensor操作"""
    op_id: str
    op_type: str  # 'reshape', 'view', 'split', 'cat', 'permute', 'transpose'
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    parameters: Dict[str, Any]
    source_layer: str
    target_layer: Optional[str]

@dataclass
class DataFlowPath:
    """数据流路径"""
    path_id: str
    start_layer: str
    end_layer: str
    operations: List[TensorOperation]
    shape_transformations: List[Tuple[int, ...]]
    
@dataclass
class TensorRoutingInfo:
    """Tensor路由信息"""
    routing_id: str
    input_tensor: str
    output_tensors: List[str]
    routing_type: str  # 'split', 'broadcast', 'gather', 'scatter'
    routing_parameters: Dict[str, Any]

@dataclass
class TensorFlowAnalysis:
    """Tensor流分析结果"""
    tensor_operations: List[TensorOperation]
    data_flow_paths: List[DataFlowPath]
    routing_info: List[TensorRoutingInfo]
    shape_changes: Dict[str, List[Tuple[int, ...]]]
    data_dependencies: Dict[str, List[str]]

class TensorFlowAnalyzer:
    """Tensor流分析器"""
    
    def __init__(self):
        # Tensor操作模式
        self.tensor_ops_patterns = {
            'reshape': [
                r'\.view\(',
                r'\.reshape\(',
                r'torch\.reshape\(',
                r'\.contiguous\(\)\.view\(',
            ],
            'split': [
                r'\.split\(',
                r'\.chunk\(',
                r'torch\.split\(',
                r'torch\.chunk\(',
                r'torch\.unbind\(',
            ],
            'concat': [
                r'torch\.cat\(',
                r'torch\.stack\(',
                r'torch\.concat\(',
                r'cat\(',
                r'stack\(',
            ],
            'permute': [
                r'\.permute\(',
                r'\.transpose\(',
                r'torch\.transpose\(',
                r'torch\.permute\(',
            ],
            'squeeze': [
                r'\.squeeze\(',
                r'\.unsqueeze\(',
                r'torch\.squeeze\(',
                r'torch\.unsqueeze\(',
            ],
            'indexing': [
                r'\[.*\]',
                r'\.index_select\(',
                r'\.gather\(',
                r'\.scatter\(',
            ]
        }
        
        # 数据流模式
        self.flow_patterns = {
            'assignment': r'(\w+)\s*=\s*(.+)',
            'function_call': r'(\w+)\s*=\s*(\w+)\(',
            'method_call': r'(\w+)\s*=\s*(\w+)\.(\w+)\(',
        }
    
    def analyze_tensor_flow(self, model: nn.Module, 
                           code: str = None,
                           input_shape: Tuple[int, ...] = None) -> TensorFlowAnalysis:
        """分析tensor流"""
        if input_shape is None:
            input_shape = (1, 3, 224, 224)
        
        # 分析tensor操作
        tensor_ops = self._analyze_tensor_operations(model, code, input_shape)
        
        # 分析数据流路径
        flow_paths = self._analyze_data_flow_paths(model, code, input_shape)
        
        # 分析tensor路由
        routing_info = self._analyze_tensor_routing(model, code)
        
        # 分析形状变化
        shape_changes = self._analyze_shape_changes(model, input_shape)
        
        # 分析数据依赖
        dependencies = self._analyze_data_dependencies(code)
        
        return TensorFlowAnalysis(
            tensor_operations=tensor_ops,
            data_flow_paths=flow_paths,
            routing_info=routing_info,
            shape_changes=shape_changes,
            data_dependencies=dependencies
        )
    
    def _analyze_tensor_operations(self, model: nn.Module, 
                                 code: str = None,
                                 input_shape: Tuple[int, ...] = None) -> List[TensorOperation]:
        """分析tensor操作"""
        operations = []
        
        # 从代码分析
        if code:
            operations.extend(self._extract_ops_from_code(code))
        
        # 从模型执行分析
        operations.extend(self._extract_ops_from_execution(model, input_shape))
        
        return operations
    
    def _extract_ops_from_code(self, code: str) -> List[TensorOperation]:
        """从代码提取操作"""
        operations = []
        op_id = 0
        
        for op_type, patterns in self.tensor_ops_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    op_id += 1
                    
                    operations.append(TensorOperation(
                        op_id=f'{op_type}_{op_id}',
                        op_type=op_type,
                        input_shapes=[],  # 需要运行时确定
                        output_shapes=[],  # 需要运行时确定
                        parameters=self._extract_op_parameters(match.group(0)),
                        source_layer='unknown',
                        target_layer=None
                    ))
        
        return operations
    
    def _extract_op_parameters(self, op_str: str) -> Dict[str, Any]:
        """提取操作参数"""
        params = {}
        
        # 简单的参数提取
        if 'view(' in op_str:
            # 提取view的参数
            view_match = re.search(r'view\((.*)\)', op_str)
            if view_match:
                params['shape'] = view_match.group(1)
        
        elif 'split(' in op_str:
            # 提取split的参数
            split_match = re.search(r'split\((.*)\)', op_str)
            if split_match:
                params['split_args'] = split_match.group(1)
        
        elif 'cat(' in op_str:
            # 提取cat的参数
            cat_match = re.search(r'cat\((.*)\)', op_str)
            if cat_match:
                params['cat_args'] = cat_match.group(1)
        
        return params
    
    def _extract_ops_from_execution(self, model: nn.Module, 
                                   input_shape: Tuple[int, ...]) -> List[TensorOperation]:
        """从模型执行提取操作"""
        operations = []
        op_counter = 0
        
        # 记录tensor操作的hook
        def tensor_op_hook(name):
            def hook(module, input, output):
                nonlocal op_counter
                op_counter += 1
                
                input_shapes = []
                output_shapes = []
                
                # 提取输入形状
                if isinstance(input, tuple):
                    for inp in input:
                        if hasattr(inp, 'shape'):
                            input_shapes.append(tuple(inp.shape))
                elif hasattr(input, 'shape'):
                    input_shapes.append(tuple(input.shape))
                
                # 提取输出形状
                if isinstance(output, tuple):
                    for out in output:
                        if hasattr(out, 'shape'):
                            output_shapes.append(tuple(out.shape))
                elif hasattr(output, 'shape'):
                    output_shapes.append(tuple(output.shape))
                
                # 判断操作类型
                op_type = self._infer_op_type(module, input_shapes, output_shapes)
                
                if op_type:
                    operations.append(TensorOperation(
                        op_id=f'exec_{op_type}_{op_counter}',
                        op_type=op_type,
                        input_shapes=input_shapes,
                        output_shapes=output_shapes,
                        parameters={'module_type': type(module).__name__},
                        source_layer=name,
                        target_layer=None
                    ))
            
            return hook
        
        # 注册hooks
        handles = []
        for name, module in model.named_modules():
            if name:
                handle = module.register_forward_hook(tensor_op_hook(name))
                handles.append(handle)
        
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(*input_shape)
                model(dummy_input)
        except Exception as e:
            print(f"执行分析失败: {e}")
        finally:
            # 清理hooks
            for handle in handles:
                handle.remove()
        
        return operations
    
    def _infer_op_type(self, module: nn.Module, 
                      input_shapes: List[Tuple], 
                      output_shapes: List[Tuple]) -> Optional[str]:
        """推断操作类型"""
        module_type = type(module).__name__
        
        # 基于模块类型推断
        if 'Conv' in module_type:
            return 'convolution'
        elif 'Linear' in module_type:
            return 'linear'
        elif 'Pool' in module_type:
            return 'pooling'
        elif 'BatchNorm' in module_type:
            return 'normalization'
        elif 'Dropout' in module_type:
            return 'dropout'
        
        # 基于形状变化推断
        if len(input_shapes) == 1 and len(output_shapes) == 1:
            input_shape = input_shapes[0]
            output_shape = output_shapes[0]
            
            if len(input_shape) != len(output_shape):
                return 'reshape'
            elif input_shape != output_shape:
                return 'transform'
        
        elif len(input_shapes) > 1 and len(output_shapes) == 1:
            return 'merge'
        elif len(input_shapes) == 1 and len(output_shapes) > 1:
            return 'split'
        
        return None
    
    def _analyze_data_flow_paths(self, model: nn.Module, 
                               code: str = None,
                               input_shape: Tuple[int, ...] = None) -> List[DataFlowPath]:
        """分析数据流路径"""
        paths = []
        
        # 通过执行追踪数据流
        if model:
            paths.extend(self._trace_execution_flow(model, input_shape))
        
        # 通过代码分析数据流
        if code:
            paths.extend(self._analyze_code_flow(code))
        
        return paths
    
    def _trace_execution_flow(self, model: nn.Module, 
                            input_shape: Tuple[int, ...]) -> List[DataFlowPath]:
        """追踪执行流"""
        paths = []
        execution_trace = []
        shape_trace = []
        
        def flow_hook(name):
            def hook(module, input, output):
                execution_trace.append(name)
                
                # 记录形状变化
                if hasattr(output, 'shape'):
                    shape_trace.append(tuple(output.shape))
                elif isinstance(output, tuple) and len(output) > 0 and hasattr(output[0], 'shape'):
                    shape_trace.append(tuple(output[0].shape))
                else:
                    shape_trace.append(None)
            
            return hook
        
        # 注册hooks
        handles = []
        for name, module in model.named_modules():
            if name:
                handle = module.register_forward_hook(flow_hook(name))
                handles.append(handle)
        
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(*input_shape)
                model(dummy_input)
            
            # 构建数据流路径
            if execution_trace:
                paths.append(DataFlowPath(
                    path_id='main_flow',
                    start_layer=execution_trace[0],
                    end_layer=execution_trace[-1],
                    operations=[],  # 详细操作需要进一步分析
                    shape_transformations=[s for s in shape_trace if s is not None]
                ))
        
        except Exception as e:
            print(f"数据流追踪失败: {e}")
        finally:
            # 清理hooks
            for handle in handles:
                handle.remove()
        
        return paths
    
    def _analyze_code_flow(self, code: str) -> List[DataFlowPath]:
        """分析代码中的数据流"""
        paths = []
        
        # 使用AST分析数据流
        try:
            tree = ast.parse(code)
            visitor = DataFlowVisitor()
            visitor.visit(tree)
            
            # 从AST访问器结果构建路径
            for i, flow in enumerate(visitor.data_flows):
                paths.append(DataFlowPath(
                    path_id=f'code_flow_{i}',
                    start_layer=flow.get('start', 'input'),
                    end_layer=flow.get('end', 'output'),
                    operations=[],
                    shape_transformations=[]
                ))
        
        except Exception as e:
            print(f"代码流分析失败: {e}")
        
        return paths
    
    def _analyze_tensor_routing(self, model: nn.Module, code: str = None) -> List[TensorRoutingInfo]:
        """分析tensor路由"""
        routing_info = []
        
        if code:
            routing_info.extend(self._extract_routing_from_code(code))
        
        return routing_info
    
    def _extract_routing_from_code(self, code: str) -> List[TensorRoutingInfo]:
        """从代码提取路由信息"""
        routing_info = []
        routing_id = 0
        
        # 检测split操作
        split_patterns = [r'torch\.split\(', r'\.split\(', r'torch\.chunk\(', r'\.chunk\(']
        for pattern in split_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                routing_id += 1
                
                routing_info.append(TensorRoutingInfo(
                    routing_id=f'split_{routing_id}',
                    input_tensor='input',
                    output_tensors=['output1', 'output2'],  # 简化
                    routing_type='split',
                    routing_parameters={'pattern': pattern}
                ))
        
        # 检测cat操作
        cat_patterns = [r'torch\.cat\(', r'torch\.stack\(']
        for pattern in cat_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                routing_id += 1
                
                routing_info.append(TensorRoutingInfo(
                    routing_id=f'merge_{routing_id}',
                    input_tensor='multiple_inputs',
                    output_tensors=['merged_output'],
                    routing_type='merge',
                    routing_parameters={'pattern': pattern}
                ))
        
        return routing_info
    
    def _analyze_shape_changes(self, model: nn.Module, 
                             input_shape: Tuple[int, ...]) -> Dict[str, List[Tuple[int, ...]]]:
        """分析形状变化"""
        shape_changes = {}
        
        def shape_hook(name):
            def hook(module, input, output):
                shapes = []
                
                # 输入形状
                if isinstance(input, tuple):
                    for inp in input:
                        if hasattr(inp, 'shape'):
                            shapes.append(('input', tuple(inp.shape)))
                elif hasattr(input, 'shape'):
                    shapes.append(('input', tuple(input.shape)))
                
                # 输出形状
                if isinstance(output, tuple):
                    for out in output:
                        if hasattr(out, 'shape'):
                            shapes.append(('output', tuple(out.shape)))
                elif hasattr(output, 'shape'):
                    shapes.append(('output', tuple(output.shape)))
                
                shape_changes[name] = shapes
            
            return hook
        
        # 注册hooks
        handles = []
        for name, module in model.named_modules():
            if name:
                handle = module.register_forward_hook(shape_hook(name))
                handles.append(handle)
        
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(*input_shape)
                model(dummy_input)
        except Exception as e:
            print(f"形状分析失败: {e}")
        finally:
            # 清理hooks
            for handle in handles:
                handle.remove()
        
        return shape_changes
    
    def _analyze_data_dependencies(self, code: str = None) -> Dict[str, List[str]]:
        """分析数据依赖关系"""
        dependencies = {}
        
        if not code:
            return dependencies
        
        # 简单的变量依赖分析
        lines = code.split('\n')
        for line in lines:
            # 查找赋值语句
            assignment_match = re.match(r'\s*(\w+)\s*=\s*(.+)', line)
            if assignment_match:
                var_name = assignment_match.group(1)
                expression = assignment_match.group(2)
                
                # 查找表达式中使用的变量
                used_vars = re.findall(r'\b(\w+)\b', expression)
                # 过滤掉函数名和关键字
                used_vars = [v for v in used_vars if v not in ['torch', 'nn', 'F', 'self']]
                
                dependencies[var_name] = used_vars
        
        return dependencies
    
    def get_tensor_flow_visualization_data(self, analysis: TensorFlowAnalysis) -> Dict[str, Any]:
        """获取tensor流的可视化数据"""
        viz_data = {
            'tensor_operations': [],
            'data_flow_paths': [],
            'routing_info': [],
            'shape_changes': analysis.shape_changes,
            'dependencies': analysis.data_dependencies
        }
        
        # Tensor操作
        for op in analysis.tensor_operations:
            viz_data['tensor_operations'].append({
                'id': op.op_id,
                'type': op.op_type,
                'input_shapes': op.input_shapes,
                'output_shapes': op.output_shapes,
                'parameters': op.parameters,
                'source': op.source_layer,
                'target': op.target_layer
            })
        
        # 数据流路径
        for path in analysis.data_flow_paths:
            viz_data['data_flow_paths'].append({
                'id': path.path_id,
                'start': path.start_layer,
                'end': path.end_layer,
                'shape_sequence': path.shape_transformations,
                'operation_count': len(path.operations)
            })
        
        # 路由信息
        for routing in analysis.routing_info:
            viz_data['routing_info'].append({
                'id': routing.routing_id,
                'input': routing.input_tensor,
                'outputs': routing.output_tensors,
                'type': routing.routing_type,
                'params': routing.routing_parameters
            })
        
        return viz_data

class DataFlowVisitor(ast.NodeVisitor):
    """数据流AST访问器"""
    
    def __init__(self):
        self.data_flows = []
        self.current_flow = {}
    
    def visit_Assign(self, node: ast.Assign):
        """访问赋值语句"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.current_flow['target'] = target.id
                
                # 分析右侧表达式
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Attribute):
                        self.current_flow['operation'] = node.value.func.attr
                        if isinstance(node.value.func.value, ast.Name):
                            self.current_flow['source'] = node.value.func.value.id
                
                self.data_flows.append(self.current_flow.copy())
                self.current_flow = {}
        
        self.generic_visit(node)
