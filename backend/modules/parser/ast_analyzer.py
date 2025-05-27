"""
AST分析器
解析Python源代码，识别PyTorch模型结构
"""

import ast
import inspect
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LayerInfo:
    """层信息数据类"""
    name: str
    layer_type: str
    args: List[Any]
    kwargs: Dict[str, Any]
    line_number: int
    
    
@dataclass
class ModelInfo:
    """模型信息数据类"""
    class_name: str
    layers: List[LayerInfo]
    forward_flow: List[str]
    imports: List[str]
    parent_classes: List[str]


class ASTAnalyzer:
    """Python AST代码分析器"""
    
    def __init__(self):
        self.pytorch_layers = {
            # 卷积层
            'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
            # 线性层
            'Linear', 'Bilinear',
            # 激活函数
            'ReLU', 'ReLU6', 'ELU', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
            'Hardswish', 'Hardtanh', 'Hardsigmoid', 'GELU', 'SiLU', 'Mish',
            # 池化层
            'MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
            'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
            'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
            # 归一化层
            'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
            # Dropout层
            'Dropout', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout',
            # 循环层
            'RNN', 'LSTM', 'GRU', 'RNNCell', 'LSTMCell', 'GRUCell',
            # Transformer层
            'Transformer', 'TransformerEncoder', 'TransformerDecoder', 'TransformerEncoderLayer', 'TransformerDecoderLayer',
            'MultiheadAttention',
            # 容器层
            'Sequential', 'ModuleList', 'ModuleDict', 'ParameterList', 'ParameterDict',
            # 其他层
            'Embedding', 'EmbeddingBag', 'Flatten', 'Unflatten'
        }
    
    def analyze_code(self, code: str) -> ModelInfo:
        """分析Python代码，提取模型信息"""
        try:
            tree = ast.parse(code)
            return self._extract_model_info(tree)
        except SyntaxError as e:
            raise ValueError(f"代码语法错误: {e}")
    
    def analyze_file(self, file_path: str) -> ModelInfo:
        """分析Python文件，提取模型信息"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            return self.analyze_code(code)
        except FileNotFoundError:
            raise ValueError(f"文件不存在: {file_path}")
        except Exception as e:
            raise ValueError(f"读取文件失败: {e}")
    
    def _extract_model_info(self, tree: ast.AST) -> ModelInfo:
        """从AST树中提取模型信息"""
        visitor = ModelVisitor(self.pytorch_layers)
        visitor.visit(tree)
        
        if not visitor.model_classes:
            raise ValueError("未找到继承自nn.Module的模型类")
        
        # 取第一个找到的模型类
        model_class = visitor.model_classes[0]
        
        return ModelInfo(
            class_name=model_class['name'],
            layers=model_class['layers'],
            forward_flow=model_class['forward_flow'],
            imports=visitor.imports,
            parent_classes=model_class['parent_classes']
        )


class ModelVisitor(ast.NodeVisitor):
    """模型类访问器"""
    
    def __init__(self, pytorch_layers: set):
        self.pytorch_layers = pytorch_layers
        self.imports = []
        self.model_classes = []
        self.current_class = None
        self.in_init = False
        self.in_forward = False
    
    def visit_Import(self, node: ast.Import):
        """访问import语句"""
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """访问from...import语句"""
        if node.module:
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                self.imports.append(full_name)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """访问类定义"""
        # 检查是否继承自nn.Module
        is_model_class = False
        parent_classes = []
        
        for base in node.bases:
            if isinstance(base, ast.Attribute):
                if isinstance(base.value, ast.Name) and base.value.id == 'nn' and base.attr == 'Module':
                    is_model_class = True
                    parent_classes.append('nn.Module')
            elif isinstance(base, ast.Name):
                parent_classes.append(base.id)
                if base.id in ['Module', 'nn.Module']:
                    is_model_class = True
        
        if is_model_class:
            self.current_class = {
                'name': node.name,
                'layers': [],
                'forward_flow': [],
                'parent_classes': parent_classes
            }
            self.generic_visit(node)
            self.model_classes.append(self.current_class)
            self.current_class = None
        else:
            self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """访问函数定义"""
        if self.current_class:
            if node.name == '__init__':
                self.in_init = True
                self.generic_visit(node)
                self.in_init = False
            elif node.name == 'forward':
                self.in_forward = True
                self.generic_visit(node)
                self.in_forward = False
        else:
            self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        """访问赋值语句"""
        if self.in_init and self.current_class:
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                    layer_name = target.attr
                    layer_info = self._extract_layer_info(layer_name, node.value, node.lineno)
                    if layer_info:
                        self.current_class['layers'].append(layer_info)
        
        if self.in_forward and self.current_class:
            # 记录前向传播中的变量赋值
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if hasattr(ast, 'unparse'):
                        value_str = ast.unparse(node.value)
                    else:
                        value_str = repr(node.value)
                    self.current_class['forward_flow'].append(f"{target.id} = {value_str}")
        
        self.generic_visit(node)
    
    def _extract_layer_info(self, layer_name: str, value_node: ast.AST, line_number: int) -> Optional[LayerInfo]:
        """提取层信息"""
        if isinstance(value_node, ast.Call):
            # 获取层类型
            layer_type = self._get_layer_type(value_node.func)
            
            if layer_type and layer_type in self.pytorch_layers:
                # 提取参数
                args = []
                kwargs = {}
                
                for arg in value_node.args:
                    args.append(self._extract_value(arg))
                
                for keyword in value_node.keywords:
                    kwargs[keyword.arg] = self._extract_value(keyword.value)
                
                return LayerInfo(
                    name=layer_name,
                    layer_type=layer_type,
                    args=args,
                    kwargs=kwargs,
                    line_number=line_number
                )
        
        return None
    
    def _get_layer_type(self, func_node: ast.AST) -> Optional[str]:
        """获取层类型名称"""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            if isinstance(func_node.value, ast.Name) and func_node.value.id == 'nn':
                return func_node.attr
            elif isinstance(func_node.value, ast.Attribute):
                # 处理torch.nn.xxx的情况
                return func_node.attr
        return None
    
    def _extract_value(self, node: ast.AST) -> Any:
        """提取节点值"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8兼容性
            return node.n
        elif isinstance(node, ast.Str):  # Python < 3.8兼容性
            return node.s
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return [self._extract_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._extract_value(elt) for elt in node.elts)
        elif isinstance(node, ast.Dict):
            return {
                self._extract_value(k): self._extract_value(v) 
                for k, v in zip(node.keys, node.values)
            }
        else:
            # 对于复杂表达式，返回其字符串表示
            try:
                # Python 3.9+的ast.unparse
                if hasattr(ast, 'unparse'):
                    return ast.unparse(node)
                else:
                    # 兼容性fallback
                    return repr(node)
            except:
                return str(node)
