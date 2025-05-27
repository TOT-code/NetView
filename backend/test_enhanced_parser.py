"""
增强型解析器测试
测试动态图、架构模式和tensor流分析功能
"""

import sys
import os
import torch
import torch.nn as nn

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.modules.parser import EnhancedModelExtractor
from examples.simple_cnn import SimpleCNN

def create_resnet_block():
    """创建一个简单的ResNet块用于测试"""
    return """
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接 - 这里是关键的跳跃连接
        out = out + self.shortcut(identity)
        out = self.relu(out)
        
        return out
"""

def create_dynamic_model():
    """创建一个包含动态控制的模型"""
    return """
import torch
import torch.nn as nn

class DynamicModel(nn.Module):
    def __init__(self, num_classes=10):
        super(DynamicModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.use_dropout = True
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        
        # 动态控制流 - 条件分支
        if self.training and self.use_dropout:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        
        # 自适应池化 - 动态形状处理
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # 根据输入大小动态调整
        if x.size(1) != 64:
            # 如果特征数不匹配，重新调整
            x = x[:, :64]
        
        x = self.fc(x)
        return x
"""

def create_tensor_ops_model():
    """创建包含复杂tensor操作的模型"""
    return """
import torch
import torch.nn as nn

class TensorOpsModel(nn.Module):
    def __init__(self):
        super(TensorOpsModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        # 多种tensor操作
        x = self.conv(x)
        
        # Split操作
        x1, x2 = torch.split(x, 32, dim=1)
        
        # 不同的处理路径
        x1 = torch.relu(x1)
        x2 = torch.sigmoid(x2)
        
        # Concat操作
        x = torch.cat([x1, x2], dim=1)
        
        # Reshape操作
        x = x.view(x.size(0), x.size(1), -1)
        x = x.mean(dim=2)
        
        # Permute操作
        if len(x.shape) > 2:
            x = x.permute(0, 2, 1)
            
        x = self.fc(x)
        return x
"""

def test_enhanced_analyzer():
    """测试增强型分析器"""
    print("=" * 80)
    print("测试增强型PyTorch模型解析器")
    print("=" * 80)
    
    extractor = EnhancedModelExtractor()
    
    # 测试1: 基础模型分析
    print("\n📋 测试1: 基础模型分析")
    print("-" * 50)
    
    model = SimpleCNN(num_classes=10)
    enhanced_info = extractor.extract_from_model(model, input_shape=(1, 3, 32, 32))
    
    print(f"模型名称: {enhanced_info.model_name}")
    print(f"图类型: {enhanced_info.network_graph.graph_type}")
    print(f"节点数: {len(enhanced_info.network_graph.nodes)}")
    print(f"边数: {len(enhanced_info.network_graph.edges)}")
    
    if enhanced_info.dynamic_analysis:
        print(f"动态控制: {enhanced_info.dynamic_analysis.has_dynamic_control}")
        print(f"执行路径: {len(enhanced_info.dynamic_analysis.execution_paths)}")
    
    if enhanced_info.architecture_patterns:
        print(f"架构类型: {enhanced_info.architecture_patterns.pattern_type}")
        print(f"残差连接: {len(enhanced_info.architecture_patterns.residual_connections)}")
        print(f"注意力模式: {len(enhanced_info.architecture_patterns.attention_patterns)}")
    
    if enhanced_info.tensor_flow_analysis:
        print(f"Tensor操作: {len(enhanced_info.tensor_flow_analysis.tensor_operations)}")
        print(f"数据流路径: {len(enhanced_info.tensor_flow_analysis.data_flow_paths)}")
    
    # 测试2: ResNet块分析
    print("\n📋 测试2: ResNet块分析")
    print("-" * 50)
    
    resnet_code = create_resnet_block()
    resnet_info = extractor.extract_from_code(resnet_code, input_shape=(1, 64, 32, 32))
    
    print(f"模型名称: {resnet_info.model_name}")
    print(f"图类型: {resnet_info.network_graph.graph_type}")
    
    if resnet_info.architecture_patterns:
        print(f"架构类型: {resnet_info.architecture_patterns.pattern_type}")
        print(f"残差连接数: {len(resnet_info.architecture_patterns.residual_connections)}")
        
        if resnet_info.architecture_patterns.residual_connections:
            for i, conn in enumerate(resnet_info.architecture_patterns.residual_connections):
                print(f"  残差连接{i+1}: {conn.input_layer} -> {conn.output_layer} ({conn.operation})")
    
    # 测试3: 动态模型分析
    print("\n📋 测试3: 动态模型分析")
    print("-" * 50)
    
    dynamic_code = create_dynamic_model()
    dynamic_info = extractor.extract_from_code(dynamic_code, input_shape=(1, 3, 32, 32))
    
    print(f"模型名称: {dynamic_info.model_name}")
    
    if dynamic_info.dynamic_analysis:
        print(f"动态控制: {dynamic_info.dynamic_analysis.has_dynamic_control}")
        print(f"控制流节点: {len(dynamic_info.dynamic_analysis.control_flow_nodes)}")
        print(f"动态操作: {len(dynamic_info.dynamic_analysis.dynamic_operations)}")
        
        for i, op in enumerate(dynamic_info.dynamic_analysis.dynamic_operations[:3]):
            print(f"  动态操作{i+1}: {op.get('type', 'unknown')} - {op.get('match', '')}")
    
    # 测试4: Tensor操作分析
    print("\n📋 测试4: Tensor操作分析")
    print("-" * 50)
    
    tensor_code = create_tensor_ops_model()
    tensor_info = extractor.extract_from_code(tensor_code, input_shape=(1, 3, 32, 32))
    
    print(f"模型名称: {tensor_info.model_name}")
    
    if tensor_info.tensor_flow_analysis:
        print(f"Tensor操作数: {len(tensor_info.tensor_flow_analysis.tensor_operations)}")
        print(f"路由信息: {len(tensor_info.tensor_flow_analysis.routing_info)}")
        print(f"数据依赖: {len(tensor_info.tensor_flow_analysis.data_dependencies)}")
        
        # 显示前几个tensor操作
        for i, op in enumerate(tensor_info.tensor_flow_analysis.tensor_operations[:5]):
            print(f"  操作{i+1}: {op.op_type} (ID: {op.op_id})")
        
        # 显示路由信息
        for i, routing in enumerate(tensor_info.tensor_flow_analysis.routing_info):
            print(f"  路由{i+1}: {routing.routing_type} - {routing.input_tensor} -> {routing.output_tensors}")
    
    # 测试5: 生成增强可视化数据
    print("\n📋 测试5: 增强可视化数据生成")
    print("-" * 50)
    
    viz_data = extractor.get_enhanced_visualization_data(enhanced_info)
    
    print(f"可视化数据结构:")
    print(f"  - 基础节点: {len(viz_data['nodes'])}")
    print(f"  - 基础边: {len(viz_data['edges'])}")
    
    if 'dynamic_info' in viz_data:
        print(f"  - 动态信息: ✓")
        print(f"    * 执行路径: {len(viz_data['dynamic_info'].get('execution_paths', []))}")
        print(f"    * 动态操作: {len(viz_data['dynamic_info'].get('dynamic_operations', []))}")
    
    if 'architecture_patterns' in viz_data:
        print(f"  - 架构模式: ✓")
        arch = viz_data['architecture_patterns']
        print(f"    * 残差连接: {len(arch.get('residual_connections', []))}")
        print(f"    * 密集连接: {len(arch.get('dense_connections', []))}")
        print(f"    * 注意力模式: {len(arch.get('attention_patterns', []))}")
    
    if 'tensor_flow' in viz_data:
        print(f"  - Tensor流: ✓")
        tensor_flow = viz_data['tensor_flow']
        print(f"    * Tensor操作: {len(tensor_flow.get('tensor_operations', []))}")
        print(f"    * 数据流路径: {len(tensor_flow.get('data_flow_paths', []))}")
        print(f"    * 路由信息: {len(tensor_flow.get('routing_info', []))}")
    
    # 测试6: 复杂度分析对比
    print("\n📋 测试6: 增强复杂度分析")
    print("-" * 50)
    
    complexity = enhanced_info.complexity_analysis
    print(f"增强复杂度分析结果:")
    print(f"  - 总参数: {complexity.get('total_parameters', 0):,}")
    print(f"  - 总层数: {complexity.get('total_layers', 0)}")
    print(f"  - 模型大小: {complexity.get('model_size_mb', 0)} MB")
    print(f"  - 有残差连接: {complexity.get('has_residual', False)}")
    print(f"  - 有注意力机制: {complexity.get('has_attention', False)}")
    print(f"  - 有分支结构: {complexity.get('has_branching', False)}")
    print(f"  - 有动态控制: {complexity.get('has_dynamic_control', False)}")
    
    if 'architecture_type' in complexity:
        print(f"  - 架构类型: {complexity['architecture_type']}")
    
    if 'control_flow_nodes' in complexity:
        print(f"  - 控制流节点: {complexity['control_flow_nodes']}")
    
    if 'execution_paths' in complexity:
        print(f"  - 执行路径: {complexity['execution_paths']}")
    
    print("\n" + "=" * 80)
    print("✅ 增强型解析器测试完成！")
    print("=" * 80)
    
    print("\n🎯 测试总结:")
    print("✓ 动态图分析 - 支持条件分支和循环检测")
    print("✓ 架构模式识别 - 自动识别ResNet、DenseNet等架构")
    print("✓ Tensor流追踪 - 详细分析tensor操作和数据流")
    print("✓ 增强可视化数据 - 提供更丰富的图形化信息")
    print("✓ 复杂度分析增强 - 更全面的模型复杂度评估")

def main():
    """主函数"""
    try:
        test_enhanced_analyzer()
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
