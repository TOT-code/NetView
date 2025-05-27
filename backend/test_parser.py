"""
测试PyTorch模型解析器功能
"""

import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.modules.parser import ModelExtractor
from examples.simple_cnn import SimpleCNN

def test_ast_analyzer():
    """测试AST分析器"""
    print("=" * 60)
    print("测试 AST 分析器")
    print("=" * 60)
    
    extractor = ModelExtractor()
    
    # 测试分析示例文件
    try:
        model_info = extractor.extract_from_file("examples/simple_cnn.py", input_shape=(1, 3, 32, 32))
        
        print(f"模型名称: {model_info.model_name}")
        
        if model_info.ast_info:
            print(f"AST分析成功 - 找到 {len(model_info.ast_info.layers)} 个层")
            print("\n层信息:")
            for layer in model_info.ast_info.layers:
                print(f"  - {layer.name}: {layer.layer_type}{layer.args} {layer.kwargs}")
            
            print(f"\n前向传播流程:")
            for flow in model_info.ast_info.forward_flow[:5]:  # 显示前5个
                print(f"  - {flow}")
        else:
            print("AST分析失败")
            
        if model_info.structure_info:
            print(f"\n模型内省成功 - 总参数: {model_info.structure_info.total_parameters:,}")
            print(f"模型大小: {model_info.structure_info.model_size_mb} MB")
            print(f"执行顺序: {len(model_info.structure_info.execution_order)} 个模块")
        else:
            print("模型内省失败")
        
        print(f"\n复杂度分析:")
        complexity = model_info.complexity_analysis
        print(f"  - 总层数: {complexity.get('total_layers', 0)}")
        print(f"  - 有残差连接: {complexity.get('has_residual', False)}")
        print(f"  - 有注意力机制: {complexity.get('has_attention', False)}")
        print(f"  - 有分支结构: {complexity.get('has_branching', False)}")
        
        print(f"\n网络图信息:")
        graph = model_info.network_graph
        print(f"  - 节点数: {len(graph.nodes)}")
        print(f"  - 边数: {len(graph.edges)}")
        print(f"  - 图类型: {graph.graph_type}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_pytorch_inspector():
    """测试PyTorch内省器"""
    print("\n" + "=" * 60)
    print("测试 PyTorch 内省器")
    print("=" * 60)
    
    extractor = ModelExtractor()
    
    try:
        # 创建模型实例
        model = SimpleCNN(num_classes=10)
        
        # 提取模型信息
        model_info = extractor.extract_from_model(model, input_shape=(1, 3, 32, 32))
        
        print(f"模型名称: {model_info.model_name}")
        
        if model_info.structure_info:
            structure = model_info.structure_info
            print(f"\n模型结构信息:")
            print(f"  - 总参数: {structure.total_parameters:,}")
            print(f"  - 可训练参数: {structure.trainable_parameters:,}")
            print(f"  - 模型大小: {structure.model_size_mb} MB")
            print(f"  - 模块数量: {len(structure.modules)}")
            
            # 显示层级摘要
            summary = extractor.pytorch_inspector.get_layer_summary(model, (1, 3, 32, 32))
            print(f"\n层级摘要:")
            print(summary)
            
        else:
            print("模型内省失败")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_visualization_data():
    """测试可视化数据生成"""
    print("\n" + "=" * 60)
    print("测试可视化数据生成")
    print("=" * 60)
    
    extractor = ModelExtractor()
    
    try:
        model = SimpleCNN(num_classes=10)
        model_info = extractor.extract_from_model(model, input_shape=(1, 3, 32, 32))
        
        # 获取可视化数据
        viz_data = extractor.get_visualization_data(model_info)
        
        print(f"可视化数据结构:")
        print(f"  - 节点数: {len(viz_data['nodes'])}")
        print(f"  - 边数: {len(viz_data['edges'])}")
        
        print(f"\n前5个节点:")
        for i, node in enumerate(viz_data['nodes'][:5]):
            print(f"  {i+1}. {node['id']}: {node['type']}")
            if node.get('input_shape'):
                print(f"     输入形状: {node['input_shape']}")
            if node.get('output_shape'):
                print(f"     输出形状: {node['output_shape']}")
            print(f"     参数数: {node['num_parameters']}")
        
        print(f"\n前5个连接:")
        for i, edge in enumerate(viz_data['edges'][:5]):
            print(f"  {i+1}. {edge['source']} -> {edge['target']} ({edge['connection_type']})")
        
        print(f"\n元数据:")
        metadata = viz_data['metadata']
        print(f"  - 模型名: {metadata['model_name']}")
        print(f"  - 图类型: {metadata['graph_type']}")
        print(f"  - 输入节点: {metadata['input_nodes']}")
        print(f"  - 输出节点: {metadata['output_nodes']}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_complex_model():
    """测试复杂模型解析"""
    print("\n" + "=" * 60)
    print("测试复杂模型解析（ResNet风格）")
    print("=" * 60)
    
    # 定义一个简单的ResNet块
    complex_model_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # 残差连接
        out = F.relu(out)
        
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = ResNetBlock(64, 64)
        self.layer2 = ResNetBlock(64, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
'''
    
    extractor = ModelExtractor()
    
    try:
        model_info = extractor.extract_from_code(complex_model_code, input_shape=(1, 3, 224, 224))
        
        print(f"模型名称: {model_info.model_name}")
        
        print(f"\n复杂度分析:")
        complexity = model_info.complexity_analysis
        print(f"  - 有残差连接: {complexity.get('has_residual', False)}")
        print(f"  - 有注意力机制: {complexity.get('has_attention', False)}")
        print(f"  - 有分支结构: {complexity.get('has_branching', False)}")
        
        if model_info.ast_info:
            print(f"\nAST分析找到 {len(model_info.ast_info.layers)} 个层")
        
        if model_info.structure_info:
            print(f"\n模型内省成功:")
            print(f"  - 总参数: {model_info.structure_info.total_parameters:,}")
            print(f"  - 模型大小: {model_info.structure_info.model_size_mb} MB")
        
        print(f"\n网络图类型: {model_info.network_graph.graph_type}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始测试PyTorch模型解析器...")
    
    # 运行所有测试
    test_ast_analyzer()
    test_pytorch_inspector()
    test_visualization_data()
    test_complex_model()
    
    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
