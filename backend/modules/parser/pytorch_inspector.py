"""
PyTorch模型内省器
直接分析PyTorch模型实例，提取网络结构信息
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import inspect

@dataclass
class ModuleInfo:
    """模块信息数据类"""
    name: str
    module_type: str
    parameters: Dict[str, Any]
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]
    num_parameters: int
    trainable_parameters: int
    children: List[str]
    parent: Optional[str]


@dataclass
class ModelStructure:
    """模型结构数据类"""
    model_name: str
    modules: Dict[str, ModuleInfo]
    execution_order: List[str]
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float


class PyTorchInspector:
    """PyTorch模型内省器"""
    
    def __init__(self):
        self.hooks = []
        self.activation_shapes = {}
    
    def inspect_model(self, model: nn.Module, input_shape: Tuple[int, ...] = None) -> ModelStructure:
        """检查模型结构"""
        if input_shape is None:
            input_shape = (1, 3, 224, 224)  # 默认输入形状
        
        # 获取模型基本信息
        model_name = model.__class__.__name__
        modules = self._extract_modules(model)
        
        # 计算参数数量
        total_params, trainable_params = self._count_parameters(model)
        model_size = self._calculate_model_size(model)
        
        # 获取执行顺序
        execution_order = self._get_execution_order(model, input_shape)
        
        # 推断形状信息
        self._infer_shapes(model, input_shape, modules)
        
        return ModelStructure(
            model_name=model_name,
            modules=modules,
            execution_order=execution_order,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=model_size
        )
    
    def _extract_modules(self, model: nn.Module) -> Dict[str, ModuleInfo]:
        """提取所有模块信息"""
        modules = {}
        
        for name, module in model.named_modules():
            if name == '':  # 跳过根模块
                continue
                
            # 获取模块参数
            module_params = self._get_module_parameters(module)
            
            # 计算参数数量
            num_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # 获取子模块
            children = [child_name for child_name, _ in module.named_children()]
            
            # 获取父模块
            parent = '.'.join(name.split('.')[:-1]) if '.' in name else None
            
            modules[name] = ModuleInfo(
                name=name,
                module_type=module.__class__.__name__,
                parameters=module_params,
                input_shape=None,  # 稍后推断
                output_shape=None,  # 稍后推断
                num_parameters=num_params,
                trainable_parameters=trainable_params,
                children=children,
                parent=parent
            )
        
        return modules
    
    def _get_module_parameters(self, module: nn.Module) -> Dict[str, Any]:
        """获取模块参数"""
        params = {}
        
        # 获取模块的构造参数
        if hasattr(module, '__dict__'):
            for key, value in module.__dict__.items():
                if not key.startswith('_') and not callable(value):
                    if isinstance(value, (int, float, str, bool, tuple, list)):
                        params[key] = value
                    elif isinstance(value, torch.Tensor) and value.numel() <= 10:
                        params[key] = value.tolist()
        
        # 特殊处理常见层类型
        if isinstance(module, nn.Conv2d):
            params.update({
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
                'dilation': module.dilation,
                'groups': module.groups,
                'bias': module.bias is not None
            })
        elif isinstance(module, nn.Linear):
            params.update({
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None
            })
        elif isinstance(module, nn.BatchNorm2d):
            params.update({
                'num_features': module.num_features,
                'eps': module.eps,
                'momentum': module.momentum,
                'affine': module.affine,
                'track_running_stats': module.track_running_stats
            })
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            params.update({
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding
            })
        elif isinstance(module, nn.Dropout):
            params.update({
                'p': module.p,
                'inplace': module.inplace
            })
        
        return params
    
    def _count_parameters(self, model: nn.Module) -> Tuple[int, int]:
        """计算模型参数数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """计算模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return round(size_mb, 2)
    
    def _get_execution_order(self, model: nn.Module, input_shape: Tuple[int, ...]) -> List[str]:
        """获取模块执行顺序"""
        execution_order = []
        
        def hook_fn(module, input, output):
            module_name = None
            for name, mod in model.named_modules():
                if mod is module:
                    module_name = name
                    break
            if module_name and module_name not in execution_order:
                execution_order.append(module_name)
        
        # 注册钩子
        handles = []
        for name, module in model.named_modules():
            if name != '':  # 跳过根模块
                handle = module.register_forward_hook(hook_fn)
                handles.append(handle)
        
        # 前向传播
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(*input_shape)
                model(dummy_input)
        except Exception as e:
            print(f"前向传播失败: {e}")
        finally:
            # 移除钩子
            for handle in handles:
                handle.remove()
        
        return execution_order
    
    def _infer_shapes(self, model: nn.Module, input_shape: Tuple[int, ...], modules: Dict[str, ModuleInfo]):
        """推断各层的输入输出形状"""
        shapes = {}
        
        def shape_hook(name):
            def hook_fn(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    input_shape = input[0].shape
                else:
                    input_shape = input.shape if hasattr(input, 'shape') else None
                
                if isinstance(output, tuple):
                    output_shape = output[0].shape if len(output) > 0 else None
                else:
                    output_shape = output.shape if hasattr(output, 'shape') else None
                
                shapes[name] = {
                    'input_shape': tuple(input_shape) if input_shape is not None else None,
                    'output_shape': tuple(output_shape) if output_shape is not None else None
                }
            return hook_fn
        
        # 注册形状钩子
        handles = []
        for name, module in model.named_modules():
            if name != '':
                handle = module.register_forward_hook(shape_hook(name))
                handles.append(handle)
        
        # 前向传播获取形状
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(*input_shape)
                model(dummy_input)
        except Exception as e:
            print(f"形状推断失败: {e}")
        finally:
            # 移除钩子
            for handle in handles:
                handle.remove()
        
        # 更新模块信息
        for name, module_info in modules.items():
            if name in shapes:
                module_info.input_shape = shapes[name]['input_shape']
                module_info.output_shape = shapes[name]['output_shape']
    
    def get_layer_summary(self, model: nn.Module, input_shape: Tuple[int, ...] = None) -> str:
        """获取层级摘要信息"""
        if input_shape is None:
            input_shape = (1, 3, 224, 224)
        
        structure = self.inspect_model(model, input_shape)
        
        summary = f"模型: {structure.model_name}\n"
        summary += "=" * 60 + "\n"
        summary += f"{'层名称':<20} {'层类型':<15} {'输出形状':<15} {'参数数量':<10}\n"
        summary += "-" * 60 + "\n"
        
        for name, module_info in structure.modules.items():
            if not module_info.children:  # 只显示叶子模块
                output_shape = str(module_info.output_shape) if module_info.output_shape else "N/A"
                summary += f"{name:<20} {module_info.module_type:<15} {output_shape:<15} {module_info.num_parameters:<10}\n"
        
        summary += "=" * 60 + "\n"
        summary += f"总参数数量: {structure.total_parameters:,}\n"
        summary += f"可训练参数: {structure.trainable_parameters:,}\n"
        summary += f"模型大小: {structure.model_size_mb} MB\n"
        
        return summary
    
    def analyze_complexity(self, model: nn.Module, input_shape: Tuple[int, ...] = None) -> Dict[str, Any]:
        """分析模型复杂度"""
        if input_shape is None:
            input_shape = (1, 3, 224, 224)
        
        structure = self.inspect_model(model, input_shape)
        
        # 统计层类型
        layer_types = {}
        for module_info in structure.modules.values():
            layer_type = module_info.module_type
            if layer_type in layer_types:
                layer_types[layer_type] += 1
            else:
                layer_types[layer_type] = 1
        
        # 计算深度
        max_depth = 0
        for name in structure.modules.keys():
            depth = name.count('.')
            max_depth = max(max_depth, depth)
        
        return {
            'total_modules': len(structure.modules),
            'layer_types': layer_types,
            'max_depth': max_depth,
            'total_parameters': structure.total_parameters,
            'trainable_parameters': structure.trainable_parameters,
            'model_size_mb': structure.model_size_mb
        }
