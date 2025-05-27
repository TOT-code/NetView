"""
动态图分析器
支持torch.jit.script分析，处理条件语句和循环结构
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import re
import ast

@dataclass
class ControlFlowNode:
    """控制流节点"""
    node_id: str
    node_type: str  # 'if', 'for', 'while', 'block'
    condition: Optional[str]
    true_branch: Optional[List[str]]
    false_branch: Optional[List[str]]
    loop_body: Optional[List[str]]
    
@dataclass
class DynamicPath:
    """动态执行路径"""
    path_id: str
    condition: str
    nodes: List[str]
    probability: float
    
@dataclass
class DynamicGraphInfo:
    """动态图信息"""
    has_dynamic_control: bool
    control_flow_nodes: List[ControlFlowNode]
    execution_paths: List[DynamicPath]
    conditional_shapes: Dict[str, List[Tuple]]
    dynamic_operations: List[Dict[str, Any]]

class DynamicGraphAnalyzer:
    """动态图分析器"""
    
    def __init__(self):
        self.jit_patterns = {
            'conditional': [
                r'if\s+.*:',
                r'torch\.where\(',
                r'torch\.cond\(',
                r'\.conditional\(',
            ],
            'loops': [
                r'for\s+.*\s+in\s+.*:',
                r'while\s+.*:',
                r'torch\.range\(',
                r'torch\.arange\(',
            ],
            'dynamic_shapes': [
                r'\.size\(\)',
                r'\.shape\[',
                r'torch\.tensor\(.*\)\.item\(\)',
                r'\.numel\(\)',
            ]
        }
    
    def analyze_dynamic_model(self, model: nn.Module, 
                            input_shape: Tuple[int, ...] = None,
                            code: str = None) -> DynamicGraphInfo:
        """分析动态模型"""
        if input_shape is None:
            input_shape = (1, 3, 224, 224)
        
        # 尝试JIT分析
        jit_info = self._analyze_with_jit(model, input_shape)
        
        # AST分析控制流
        control_flow = self._analyze_control_flow_ast(code) if code else []
        
        # 检测动态操作
        dynamic_ops = self._detect_dynamic_operations(model, code)
        
        # 分析执行路径
        execution_paths = self._analyze_execution_paths(model, input_shape, control_flow)
        
        # 检测条件形状变化
        conditional_shapes = self._analyze_conditional_shapes(model, input_shape)
        
        has_dynamic = bool(control_flow or dynamic_ops or 
                          any('dynamic' in str(op) for op in dynamic_ops))
        
        return DynamicGraphInfo(
            has_dynamic_control=has_dynamic,
            control_flow_nodes=control_flow,
            execution_paths=execution_paths,
            conditional_shapes=conditional_shapes,
            dynamic_operations=dynamic_ops
        )
    
    def _analyze_with_jit(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """使用JIT分析模型"""
        jit_info = {
            'scriptable': False,
            'traceable': False,
            'graph': None,
            'error': None
        }
        
        try:
            # 尝试torch.jit.script
            dummy_input = torch.randn(*input_shape)
            
            try:
                scripted_model = torch.jit.script(model)
                jit_info['scriptable'] = True
                jit_info['graph'] = scripted_model.graph
                print(f"JIT Script 成功")
            except Exception as script_error:
                jit_info['error'] = f"Script failed: {script_error}"
                
                # 尝试torch.jit.trace作为fallback
                try:
                    model.eval()
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model, dummy_input)
                        jit_info['traceable'] = True
                        jit_info['graph'] = traced_model.graph
                        print(f"JIT Trace 成功")
                except Exception as trace_error:
                    jit_info['error'] += f", Trace failed: {trace_error}"
                    print(f"JIT 分析失败: {jit_info['error']}")
        
        except Exception as e:
            jit_info['error'] = f"JIT analysis failed: {e}"
            print(f"JIT 分析出错: {e}")
        
        return jit_info
    
    def _analyze_control_flow_ast(self, code: str) -> List[ControlFlowNode]:
        """使用AST分析控制流"""
        if not code:
            return []
        
        control_flow_nodes = []
        
        try:
            tree = ast.parse(code)
            visitor = ControlFlowVisitor()
            visitor.visit(tree)
            control_flow_nodes = visitor.control_nodes
        except Exception as e:
            print(f"AST控制流分析失败: {e}")
        
        return control_flow_nodes
    
    def _detect_dynamic_operations(self, model: nn.Module, code: str = None) -> List[Dict[str, Any]]:
        """检测动态操作"""
        dynamic_ops = []
        
        # 从代码检测
        if code:
            for op_type, patterns in self.jit_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, code)
                    for match in matches:
                        dynamic_ops.append({
                            'type': op_type,
                            'pattern': pattern,
                            'match': match,
                            'source': 'code_analysis'
                        })
        
        # 从模型结构检测
        for name, module in model.named_modules():
            if hasattr(module, 'forward'):
                # 检查是否包含动态操作
                if self._has_dynamic_forward(module):
                    dynamic_ops.append({
                        'type': 'dynamic_forward',
                        'module': name,
                        'module_type': type(module).__name__,
                        'source': 'module_inspection'
                    })
        
        return dynamic_ops
    
    def _has_dynamic_forward(self, module: nn.Module) -> bool:
        """检查模块是否有动态前向传播"""
        try:
            import inspect
            source = inspect.getsource(module.forward)
            
            # 检查动态模式
            for patterns in self.jit_patterns.values():
                for pattern in patterns:
                    if re.search(pattern, source):
                        return True
        except:
            pass
        
        return False
    
    def _analyze_execution_paths(self, model: nn.Module, 
                                input_shape: Tuple[int, ...],
                                control_flow: List[ControlFlowNode]) -> List[DynamicPath]:
        """分析执行路径"""
        paths = []
        
        if not control_flow:
            # 简单的顺序执行路径
            execution_order = []
            try:
                # 获取执行顺序
                hooks = []
                
                def hook_fn(name):
                    def hook(module, input, output):
                        execution_order.append(name)
                    return hook
                
                for name, module in model.named_modules():
                    if name:
                        handle = module.register_forward_hook(hook_fn(name))
                        hooks.append(handle)
                
                model.eval()
                with torch.no_grad():
                    dummy_input = torch.randn(*input_shape)
                    model(dummy_input)
                
                # 清理hooks
                for handle in hooks:
                    handle.remove()
                
                paths.append(DynamicPath(
                    path_id='sequential',
                    condition='default',
                    nodes=execution_order,
                    probability=1.0
                ))
            
            except Exception as e:
                print(f"执行路径分析失败: {e}")
        
        else:
            # 分析条件执行路径
            for i, cf_node in enumerate(control_flow):
                if cf_node.node_type == 'if':
                    # True分支
                    if cf_node.true_branch:
                        paths.append(DynamicPath(
                            path_id=f'if_true_{i}',
                            condition=cf_node.condition or 'unknown',
                            nodes=cf_node.true_branch,
                            probability=0.5
                        ))
                    
                    # False分支
                    if cf_node.false_branch:
                        paths.append(DynamicPath(
                            path_id=f'if_false_{i}',
                            condition=f'not ({cf_node.condition})' if cf_node.condition else 'unknown',
                            nodes=cf_node.false_branch,
                            probability=0.5
                        ))
        
        return paths
    
    def _analyze_conditional_shapes(self, model: nn.Module, 
                                   input_shape: Tuple[int, ...]) -> Dict[str, List[Tuple]]:
        """分析条件形状变化"""
        conditional_shapes = {}
        
        # 尝试用不同的输入形状测试
        test_shapes = [
            input_shape,
            (input_shape[0], input_shape[1], input_shape[2]//2, input_shape[3]//2),
            (input_shape[0]*2, input_shape[1], input_shape[2], input_shape[3])
        ]
        
        for i, test_shape in enumerate(test_shapes):
            try:
                shapes = {}
                
                def shape_hook(name):
                    def hook(module, input, output):
                        if hasattr(output, 'shape'):
                            shapes[name] = tuple(output.shape)
                        elif isinstance(output, tuple) and len(output) > 0 and hasattr(output[0], 'shape'):
                            shapes[name] = tuple(output[0].shape)
                    return hook
                
                hooks = []
                for name, module in model.named_modules():
                    if name:
                        handle = module.register_forward_hook(shape_hook(name))
                        hooks.append(handle)
                
                model.eval()
                with torch.no_grad():
                    dummy_input = torch.randn(*test_shape)
                    model(dummy_input)
                
                # 清理hooks
                for handle in hooks:
                    handle.remove()
                
                conditional_shapes[f'input_shape_{i}'] = shapes
            
            except Exception as e:
                print(f"条件形状分析失败 (shape {test_shape}): {e}")
                continue
        
        return conditional_shapes
    
    def get_dynamic_visualization_data(self, dynamic_info: DynamicGraphInfo) -> Dict[str, Any]:
        """获取动态图的可视化数据"""
        viz_data = {
            'is_dynamic': dynamic_info.has_dynamic_control,
            'control_flow_nodes': [],
            'execution_paths': [],
            'dynamic_operations': dynamic_info.dynamic_operations
        }
        
        # 控制流节点
        for cf_node in dynamic_info.control_flow_nodes:
            viz_data['control_flow_nodes'].append({
                'id': cf_node.node_id,
                'type': cf_node.node_type,
                'condition': cf_node.condition,
                'branches': {
                    'true': cf_node.true_branch,
                    'false': cf_node.false_branch,
                    'loop': cf_node.loop_body
                }
            })
        
        # 执行路径
        for path in dynamic_info.execution_paths:
            viz_data['execution_paths'].append({
                'id': path.path_id,
                'condition': path.condition,
                'nodes': path.nodes,
                'probability': path.probability
            })
        
        return viz_data

class ControlFlowVisitor(ast.NodeVisitor):
    """控制流AST访问器"""
    
    def __init__(self):
        self.control_nodes = []
        self.node_counter = 0
    
    def visit_If(self, node: ast.If):
        """访问if语句"""
        self.node_counter += 1
        
        # 提取条件
        condition = self._extract_condition(node.test)
        
        # 提取分支
        true_branch = self._extract_branch_nodes(node.body)
        false_branch = self._extract_branch_nodes(node.orelse) if node.orelse else None
        
        control_node = ControlFlowNode(
            node_id=f'if_{self.node_counter}',
            node_type='if',
            condition=condition,
            true_branch=true_branch,
            false_branch=false_branch,
            loop_body=None
        )
        
        self.control_nodes.append(control_node)
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For):
        """访问for循环"""
        self.node_counter += 1
        
        # 提取循环条件
        condition = f"for {self._extract_target(node.target)} in {self._extract_iter(node.iter)}"
        
        # 提取循环体
        loop_body = self._extract_branch_nodes(node.body)
        
        control_node = ControlFlowNode(
            node_id=f'for_{self.node_counter}',
            node_type='for',
            condition=condition,
            true_branch=None,
            false_branch=None,
            loop_body=loop_body
        )
        
        self.control_nodes.append(control_node)
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While):
        """访问while循环"""
        self.node_counter += 1
        
        condition = self._extract_condition(node.test)
        loop_body = self._extract_branch_nodes(node.body)
        
        control_node = ControlFlowNode(
            node_id=f'while_{self.node_counter}',
            node_type='while',
            condition=condition,
            true_branch=None,
            false_branch=None,
            loop_body=loop_body
        )
        
        self.control_nodes.append(control_node)
        self.generic_visit(node)
    
    def _extract_condition(self, node: ast.AST) -> str:
        """提取条件表达式"""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                return repr(node)
        except:
            return 'unknown_condition'
    
    def _extract_target(self, node: ast.AST) -> str:
        """提取循环目标"""
        if isinstance(node, ast.Name):
            return node.id
        else:
            return str(node)
    
    def _extract_iter(self, node: ast.AST) -> str:
        """提取迭代器"""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                return repr(node)
        except:
            return 'unknown_iter'
    
    def _extract_branch_nodes(self, body: List[ast.stmt]) -> List[str]:
        """提取分支中的节点"""
        nodes = []
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        nodes.append(target.attr)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                # 可能是方法调用
                if isinstance(stmt.value.func, ast.Attribute):
                    nodes.append(stmt.value.func.attr)
        
        return nodes
