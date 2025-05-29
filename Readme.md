```markdown
# NetView - PyTorch模型结构可视化工具

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📖 项目简介

NetView是一个专注于PyTorch深度学习模型结构分析与可视化的工具。通过AI驱动的多层次分析技术，为研究人员和开发者提供直观的模型理解和优化建议。

### 🌟 核心特性

- **多维度模型分析**
  - AST静态代码分析
  - PyTorch运行时内省
  - 动态计算图追踪
  - Tensor流向分析

- **智能架构识别**
  - 自动识别CNN、ResNet、DenseNet等经典架构
  - 检测架构模式和设计原则
  - 提供优化建议

- **交互式可视化**
  - 基于Vis.js的网络图展示
  - 多种布局算法支持
  - 实时交互和探索

- **多格式导出**
  - JSON数据导出
  - PNG/SVG图像导出
  - 分析报告生成

## 🏗️ 项目架构
```

NetView/ ├── backend/ # 后端API服务 │ ├── api/ # FastAPI路由和服务 │ │ ├── routers/ # API路由定义 │ │ ├── services/ # 业务逻辑服务 │ │ ├── schemas/ # 数据模型定义 │ │ └── middleware/ # 中间件 │ └── modules/ # 核心分析模块 │ └── parser/ # 模型解析器 ├── frontend/ # 前端界面 │ ├── index.html # 主页面 │ ├── script.js # 交互逻辑 │ └── style.css # 样式文件 ├── examples/ # 示例模型 ├── output/ # 分析输出 └── config.py # 配置文件

````javascript

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Node.js (可选，用于前端开发)
- Git

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/你的用户名/NetView.git
   cd NetView
````

2. __安装Python依赖__

   ```bash
   pip install -r requirements-minimal.txt
   ```

3. __启动服务__

   ```bash
   # 方式1：使用启动脚本
   python start_simple.py

   # 方式2：使用批处理文件(Windows)
   start_all.bat

   # 方式3：手动启动
   cd backend
   uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
   ```

4. __访问应用__

   - 前端界面: http://localhost:8080
   - API文档: http://localhost:8001/api/docs
   - 健康检查: http://localhost:8001/api/health

### 使用示例

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 将此代码粘贴到NetView前端进行分析
```

## 📚 API文档

### 主要端点

| 端点 | 方法 | 描述 | |------|------|------| | `/api/v1/models/analyze` | POST | 分析PyTorch模型代码 | | `/api/v1/visualization/generate` | POST | 生成可视化数据 | | `/api/v1/export/image` | POST | 导出图像 | | `/api/v1/config` | GET | 获取配置信息 |

### 分析选项

```json
{
  "enable_dynamic": true,
  "enable_patterns": true,
  "enable_tensor_flow": true,
  "custom_input_shape": [1, 3, 224, 224]
}
```

## 🛠️ 开发指南

### 核心模块

1. __模型解析器__ (`backend/modules/parser/`)

   - `ast_analyzer.py`: AST静态分析
   - `pytorch_inspector.py`: PyTorch运行时分析
   - `dynamic_analyzer.py`: 动态图分析
   - `tensor_flow_analyzer.py`: Tensor流分析

2. __API服务__ (`backend/api/`)

   - FastAPI框架构建
   - RESTful API设计
   - 异步处理支持

3. __前端界面__ (`frontend/`)

   - 原生JavaScript实现
   - Vis.js网络图可视化
   - 响应式设计

### 配置说明

项目配置集中在 `config.py` 中：

- __基础配置__: 服务器端口、调试模式等
- __可视化配置__: 节点样式、颜色映射、布局选项
- __解析配置__: 超时设置、支持的层类型等
- __导出配置__: 支持格式、文件大小限制等

## 🐛 已知问题

### 前端显示问题

- __问题描述__: 当前前端显示效果不佳，用户界面需要优化

- __具体表现__:

  - 界面布局可能不够直观
  - 可视化图表显示效果有待改进
  - 用户交互体验需要提升

- __优先级__: 高

- __状态__: 待修复

- __解决方案__:

  - 重新设计UI/UX界面
  - 优化可视化图表渲染
  - 改进响应式设计
  - 添加更多交互功能

### 其他已知问题

1. __前端界面简化__ - 界面功能需要进一步完善
2. __大型模型分析性能__ - 对于复杂模型的分析速度有待优化
3. __浏览器兼容性__ - 部分旧版本浏览器可能存在兼容问题



## 🗺️ 发展路线

### v0.2.0 (当前版本)

- [x] 基础模型分析功能
- [x] FastAPI后端架构
- [x] 简单前端界面
- [x] 多格式导出

### v0.3.0 (计划中)

- [ ] 前端界面重新设计 🔥
- [ ] 性能优化和缓存
- [ ] 更多架构模式支持
- [ ] 批量分析功能

### v0.4.0 (未来)

- [ ] 机器学习模型推荐
- [ ] 云端分析服务
- [ ] 团队协作功能
- [ ] 移动端支持

