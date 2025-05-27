"""
PyTorch模型解析器使用示例
演示如何使用NetView的模型解析功能
"""

import sys
import os
import json

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.modules.parser import ModelExtractor
from examples.simple_cnn import SimpleCNN

def demo_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("PyTorch模型解析器基础使用示例")
    print("=" * 60)
    
    # 创建解析器
    extractor = ModelExtractor()
    
    # 方法1: 从模型实例解析
    print("\n1. 从模型实例解析:")
    model = SimpleCNN(num_classes=10)
    model_info = extractor.extract_from_model(model, input_shape=(1, 3, 32, 32))
    
    print(f"   模型名称: {model_info.model_name}")
    print(f"   总参数: {model_info.structure_info.total_parameters:,}")
    print(f"   模型大小: {model_info.structure_info.model_size_mb} MB")
    print(f"   图类型: {model_info.network_graph.graph_type}")
    
    # 方法2: 从文件解析
    print("\n2. 从文件解析:")
    model_info = extractor.extract_from_file("examples/simple_cnn.py", input_shape=(1, 3, 32, 32))
    
    if model_info.ast_info:
        print(f"   找到层数: {len(model_info.ast_info.layers)}")
        print(f"   前向流程步数: {len(model_info.ast_info.forward_flow)}")
    
    # 方法3: 从代码字符串解析
    print("\n3. 从代码字符串解析:")
    simple_model_code = '''
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16*30*30, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
'''
    
    model_info = extractor.extract_from_code(simple_model_code, input_shape=(1, 3, 32, 32))
    print(f"   模型名称: {model_info.model_name}")
    if model_info.ast_info:
        print(f"   解析到的层: {[layer.name for layer in model_info.ast_info.layers]}")

def demo_visualization_data():
    """可视化数据生成示例"""
    print("\n" + "=" * 60)
    print("可视化数据生成示例")
    print("=" * 60)
    
    extractor = ModelExtractor()
    model = SimpleCNN(num_classes=10)
    model_info = extractor.extract_from_model(model, input_shape=(1, 3, 32, 32))
    
    # 获取可视化数据
    viz_data = extractor.get_visualization_data(model_info)
    
    print(f"\n生成的可视化数据结构:")
    print(f"  - 节点数量: {len(viz_data['nodes'])}")
    print(f"  - 连接数量: {len(viz_data['edges'])}")
    
    # 显示节点信息
    print(f"\n节点详情:")
    for i, node in enumerate(viz_data['nodes'][:3]):  # 显示前3个节点
        print(f"  节点{i+1}: {node['id']}")
        print(f"    类型: {node['type']}")
        print(f"    标签: {node['label']}")
        if node.get('input_shape'):
            print(f"    输入形状: {node['input_shape']}")
        if node.get('output_shape'):
            print(f"    输出形状: {node['output_shape']}")
        print(f"    参数数量: {node['num_parameters']}")
        print()
    
    # 显示连接信息  
    print(f"连接详情:")
    for i, edge in enumerate(viz_data['edges'][:3]):  # 显示前3个连接
        print(f"  连接{i+1}: {edge['source']} -> {edge['target']}")
        print(f"    类型: {edge['connection_type']}")
        print(f"    数据流: {edge['data_flow']}")
        print()

def demo_model_analysis():
    """模型分析示例"""
    print("\n" + "=" * 60)
    print("模型分析功能示例")
    print("=" * 60)
    
    extractor = ModelExtractor()
    model = SimpleCNN(num_classes=10)
    
    # 复杂度分析
    complexity = extractor.pytorch_inspector.analyze_complexity(model, (1, 3, 32, 32))
    
    print(f"\n复杂度分析结果:")
    print(f"  - 模块总数: {complexity['total_modules']}")
    print(f"  - 最大深度: {complexity['max_depth']}")
    print(f"  - 总参数: {complexity['total_parameters']:,}")
    print(f"  - 可训练参数: {complexity['trainable_parameters']:,}")
    print(f"  - 模型大小: {complexity['model_size_mb']} MB")
    
    print(f"\n  层类型统计:")
    for layer_type, count in complexity['layer_types'].items():
        print(f"    {layer_type}: {count}")
    
    # 层级摘要
    print(f"\n层级摘要:")
    summary = extractor.pytorch_inspector.get_layer_summary(model, (1, 3, 32, 32))
    print(summary)

def demo_save_results():
    """保存结果示例"""
    print("\n" + "=" * 60)
    print("保存分析结果示例")
    print("=" * 60)
    
    extractor = ModelExtractor()
    model = SimpleCNN(num_classes=10)
    model_info = extractor.extract_from_model(model, input_shape=(1, 3, 32, 32))
    
    # 转换为字典格式
    model_dict = extractor.to_dict(model_info)
    
    # 获取可视化数据
    viz_data = extractor.get_visualization_data(model_info)
    
    # 保存到文件
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存完整模型信息
    with open(f"{output_dir}/model_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(model_dict, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存可视化数据
    with open(f"{output_dir}/visualization_data.json", 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"结果已保存到:")
    print(f"  - {output_dir}/model_analysis.json (完整分析结果)")
    print(f"  - {output_dir}/visualization_data.json (可视化数据)")
    
    # 显示文件大小
    model_size = os.path.getsize(f"{output_dir}/model_analysis.json")
    viz_size = os.path.getsize(f"{output_dir}/visualization_data.json")
    print(f"\n文件大小:")
    print(f"  - 模型分析: {model_size} bytes")
    print(f"  - 可视化数据: {viz_size} bytes")

def main():
    """主函数"""
    print("NetView PyTorch模型解析器演示")
    print("版本: 0.1.0")
    print("功能: PyTorch模型结构分析和可视化数据生成")
    
    try:
        demo_basic_usage()
        demo_visualization_data()
        demo_model_analysis()
        demo_save_results()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 查看生成的JSON文件了解数据格式")
        print("2. 集成到Web界面进行可视化显示")
        print("3. 测试更复杂的模型结构")
        
    except Exception as e:
        print(f"\n演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
