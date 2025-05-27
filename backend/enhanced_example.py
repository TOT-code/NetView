"""
增强型模型解析器使用示例
展示动态图、架构模式和tensor流分析的完整功能
"""

import sys
import os
import torch
import torch.nn as nn
import json

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.modules.parser import EnhancedModelExtractor
from examples.simple_cnn import SimpleCNN

def demonstrate_enhanced_features():
    """演示增强功能"""
    print("🚀 NetView 增强型模型解析器演示")
    print("版本: 0.2.0 (增强版)")
    print("新增功能: 动态图分析、架构模式识别、Tensor流追踪")
    print("=" * 80)
    
    # 创建增强型提取器
    extractor = EnhancedModelExtractor()
    
    # 1. 基础模型的增强分析
    print("\n🔍 1. 基础模型增强分析")
    print("-" * 60)
    
    model = SimpleCNN(num_classes=10)
    enhanced_info = extractor.extract_from_model(model, input_shape=(1, 3, 32, 32))
    
    print(f"✓ 模型名称: {enhanced_info.model_name}")
    print(f"✓ 架构类型: {enhanced_info.network_graph.graph_type}")
    print(f"✓ 基础结构: {len(enhanced_info.network_graph.nodes)} 节点, {len(enhanced_info.network_graph.edges)} 连接")
    
    # 动态分析结果
    if enhanced_info.dynamic_analysis:
        print(f"🔄 动态分析:")
        print(f"  - 动态控制: {enhanced_info.dynamic_analysis.has_dynamic_control}")
        print(f"  - 执行路径: {len(enhanced_info.dynamic_analysis.execution_paths)}")
        print(f"  - 动态操作: {len(enhanced_info.dynamic_analysis.dynamic_operations)}")
    
    # 架构模式结果
    if enhanced_info.architecture_patterns:
        print(f"🏗️ 架构模式:")
        patterns = enhanced_info.architecture_patterns
        print(f"  - 模式类型: {patterns.pattern_type}")
        print(f"  - 残差连接: {len(patterns.residual_connections)}")
        print(f"  - 密集连接: {len(patterns.dense_connections)}")
        print(f"  - 注意力模式: {len(patterns.attention_patterns)}")
        print(f"  - 分支点: {len(patterns.branch_points)}")
        print(f"  - 合并点: {len(patterns.merge_points)}")
    
    # Tensor流分析结果
    if enhanced_info.tensor_flow_analysis:
        print(f"📊 Tensor流分析:")
        tensor_flow = enhanced_info.tensor_flow_analysis
        print(f"  - Tensor操作: {len(tensor_flow.tensor_operations)}")
        print(f"  - 数据流路径: {len(tensor_flow.data_flow_paths)}")
        print(f"  - 路由信息: {len(tensor_flow.routing_info)}")
        print(f"  - 形状变化记录: {len(tensor_flow.shape_changes)}")
        print(f"  - 数据依赖: {len(tensor_flow.data_dependencies)}")
    
    # 2. 创建包含ResNet块的模型
    print("\n🏗️ 2. ResNet架构模式识别")
    print("-" * 60)
    
    resnet_code = """
import torch
import torch.nn as nn

class MiniResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.shortcut = nn.Conv2d(64, 128, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 第一层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 残差块
        identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 残差连接 - 关键特征
        identity = self.shortcut(identity)
        out = out + identity
        out = self.relu(out)
        
        # 分类层
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out
"""
    
    resnet_info = extractor.extract_from_code(resnet_code, input_shape=(1, 3, 32, 32))
    
    print(f"✓ 识别模型: {resnet_info.model_name}")
    print(f"✓ 架构类型: {resnet_info.network_graph.graph_type}")
    
    if resnet_info.architecture_patterns:
        patterns = resnet_info.architecture_patterns
        print(f"🎯 ResNet模式识别:")
        print(f"  - 检测到架构: {patterns.pattern_type}")
        print(f"  - 残差连接数: {len(patterns.residual_connections)}")
        
        for i, conn in enumerate(patterns.residual_connections):
            print(f"    残差连接{i+1}: {conn.input_layer} -> {conn.output_layer}")
            if conn.shortcut_layer:
                print(f"      shortcut: {conn.shortcut_layer}")
    
    # 3. 动态控制流分析
    print("\n🔄 3. 动态控制流分析")
    print("-" * 60)
    
    dynamic_code = """
import torch
import torch.nn as nn

class AdaptiveNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AdaptiveNet, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Conv2d(64, 128, 3, padding=1)
        ])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 动态选择层数
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            x = torch.relu(x)
            
            # 条件性dropout
            if self.training and i > 0:
                x = self.dropout(x)
            
            # 动态池化
            if x.size(2) > 8:  # 如果特征图太大
                x = torch.nn.functional.max_pool2d(x, 2)
        
        # 自适应处理
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # 动态调整特征维度
        if x.size(1) != 128:
            # 如果维度不匹配，进行调整
            if x.size(1) > 128:
                x = x[:, :128]
            else:
                padding = torch.zeros(x.size(0), 128 - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        x = self.classifier(x)
        return x
"""
    
    dynamic_info = extractor.extract_from_code(dynamic_code, input_shape=(1, 3, 64, 64))
    
    print(f"✓ 模型名称: {dynamic_info.model_name}")
    
    if dynamic_info.dynamic_analysis:
        dynamic = dynamic_info.dynamic_analysis
        print(f"🔄 动态特征检测:")
        print(f"  - 动态控制: {dynamic.has_dynamic_control}")
        print(f"  - 控制流节点: {len(dynamic.control_flow_nodes)}")
        print(f"  - 执行路径: {len(dynamic.execution_paths)}")
        print(f"  - 动态操作: {len(dynamic.dynamic_operations)}")
        
        # 显示检测到的动态操作
        for i, op in enumerate(dynamic.dynamic_operations[:5]):
            op_type = op.get('type', 'unknown')
            match = op.get('match', '')
            print(f"    动态操作{i+1}: {op_type} - {match[:50]}...")
    
    # 4. Tensor操作追踪
    print("\n📊 4. Tensor操作流追踪")
    print("-" * 60)
    
    tensor_ops_code = """
import torch
import torch.nn as nn

class MultiPathNet(nn.Module):
    def __init__(self):
        super(MultiPathNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2a = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2b = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        # 基础卷积
        x = self.conv1(x)
        x = torch.relu(x)
        
        # 分支处理
        x1, x2 = torch.split(x, 32, dim=1)  # Split操作
        
        # 并行处理路径
        x1 = self.conv2a(x1)
        x1 = torch.relu(x1)
        
        x2 = self.conv2b(x2)
        x2 = torch.sigmoid(x2)
        
        # 合并路径
        x = torch.cat([x1, x2], dim=1)  # Concat操作
        
        # 进一步处理
        x = self.conv3(x)
        x = torch.relu(x)
        
        # 形状变换
        x = x.view(x.size(0), x.size(1), -1)  # Reshape
        x = x.mean(dim=2)  # 降维
        
        # 如果需要，进行转置
        if len(x.shape) > 2:
            x = x.transpose(1, 2)
        
        x = self.fc(x)
        return x
"""
    
    tensor_info = extractor.extract_from_code(tensor_ops_code, input_shape=(1, 3, 32, 32))
    
    print(f"✓ 模型名称: {tensor_info.model_name}")
    
    if tensor_info.tensor_flow_analysis:
        tensor_flow = tensor_info.tensor_flow_analysis
        print(f"📊 Tensor操作分析:")
        print(f"  - 总操作数: {len(tensor_flow.tensor_operations)}")
        print(f"  - 数据流路径: {len(tensor_flow.data_flow_paths)}")
        print(f"  - 路由节点: {len(tensor_flow.routing_info)}")
        
        # 显示主要操作类型
        op_types = {}
        for op in tensor_flow.tensor_operations:
            op_types[op.op_type] = op_types.get(op.op_type, 0) + 1
        
        print(f"  📋 操作类型统计:")
        for op_type, count in op_types.items():
            print(f"    {op_type}: {count}个")
        
        # 显示路由信息
        print(f"  🔀 数据路由:")
        for i, routing in enumerate(tensor_flow.routing_info):
            print(f"    路由{i+1}: {routing.routing_type}")
            print(f"      输入: {routing.input_tensor}")
            print(f"      输出: {routing.output_tensors}")
    
    # 5. 生成完整的增强可视化数据
    print("\n📈 5. 增强可视化数据生成")
    print("-" * 60)
    
    viz_data = extractor.get_enhanced_visualization_data(enhanced_info)
    
    print(f"✓ 生成增强可视化数据:")
    print(f"  📊 基础图结构:")
    print(f"    - 节点: {len(viz_data['nodes'])}")
    print(f"    - 边: {len(viz_data['edges'])}")
    print(f"    - 输入节点: {viz_data['metadata']['input_nodes']}")
    print(f"    - 输出节点: {viz_data['metadata']['output_nodes']}")
    
    if 'dynamic_info' in viz_data:
        dynamic_viz = viz_data['dynamic_info']
        print(f"  🔄 动态分析数据:")
        print(f"    - 是否动态: {dynamic_viz['is_dynamic']}")
        print(f"    - 控制流节点: {len(dynamic_viz['control_flow_nodes'])}")
        print(f"    - 执行路径: {len(dynamic_viz['execution_paths'])}")
    
    if 'architecture_patterns' in viz_data:
        arch_viz = viz_data['architecture_patterns']
        print(f"  🏗️ 架构模式数据:")
        print(f"    - 架构类型: {arch_viz['architecture_type']}")
        print(f"    - 残差连接: {len(arch_viz['residual_connections'])}")
        print(f"    - 分支点: {len(arch_viz['branch_points'])}")
        print(f"    - 合并点: {len(arch_viz['merge_points'])}")
    
    if 'tensor_flow' in viz_data:
        tensor_viz = viz_data['tensor_flow']
        print(f"  📊 Tensor流数据:")
        print(f"    - Tensor操作: {len(tensor_viz['tensor_operations'])}")
        print(f"    - 数据流路径: {len(tensor_viz['data_flow_paths'])}")
        print(f"    - 路由信息: {len(tensor_viz['routing_info'])}")
    
    # 6. 保存增强分析结果
    print("\n💾 6. 保存增强分析结果")
    print("-" * 60)
    
    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)
    
    # 保存增强分析结果
    enhanced_result = extractor.to_dict(enhanced_info)
    with open('output/enhanced_model_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(enhanced_result, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存增强可视化数据
    with open('output/enhanced_visualization_data.json', 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ 保存完成:")
    print(f"  - 完整分析结果: output/enhanced_model_analysis.json")
    print(f"  - 可视化数据: output/enhanced_visualization_data.json")
    
    # 显示文件大小
    analysis_size = os.path.getsize('output/enhanced_model_analysis.json')
    viz_size = os.path.getsize('output/enhanced_visualization_data.json')
    print(f"  - 分析文件大小: {analysis_size:,} bytes")
    print(f"  - 可视化文件大小: {viz_size:,} bytes")
    
    print("\n" + "=" * 80)
    print("🎉 增强型模型解析器演示完成！")
    print("=" * 80)
    
    print("\n🚀 增强功能总结:")
    print("✅ 动态图分析 - 检测条件分支、循环和动态形状变化")
    print("✅ 架构模式识别 - 自动识别ResNet、DenseNet、Transformer等架构")
    print("✅ Tensor流追踪 - 详细分析tensor的split、concat、reshape等操作")
    print("✅ 增强可视化 - 提供更丰富的图形化数据")
    print("✅ 智能连接检测 - 识别残差连接、注意力机制等复杂连接模式")
    print("✅ 完整数据导出 - 支持JSON格式的详细分析结果")
    
    print("\n📋 支持的新特性:")
    print("  🔄 动态控制流: if语句、for循环、while循环")
    print("  🏗️ 架构模式: ResNet、DenseNet、Transformer、U-Net")
    print("  📊 Tensor操作: view、reshape、split、cat、permute、transpose")
    print("  🔀 数据路由: split分发、concat聚合、广播、收集")
    print("  🎯 智能识别: 自动检测模型类型和复杂连接关系")

def main():
    """主函数"""
    try:
        demonstrate_enhanced_features()
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
