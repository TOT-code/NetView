# NetView - PyTorch网络结构可视化工具

## 项目概述

NetView是一个专为PyTorch深度学习模型设计的网络结构可视化工具。它能够解析PyTorch模型定义代码，并生成交互式的网络结构图，帮助开发者更好地理解和分析深度学习模型的架构。

## 主要功能

- 🔍 **自动解析PyTorch模型**：输入模型定义代码，自动提取网络结构信息
- 📊 **详细信息展示**：显示层类型、名称、参数数量和tensor形状变化
- 🖱️ **交互式可视化**：支持拖拽、缩放、点击查看详情等交互操作
- 💾 **静态图片导出**：支持导出为PNG、SVG等格式的静态图片
- 🌐 **Web界面**：基于Web的用户友好界面，无需安装额外软件

## 技术架构

### 后端技术栈
- **Python 3.8+**
- **PyTorch**：模型解析和分析
- **Flask/FastAPI**：Web API服务
- **AST模块**：Python代码静态分析

### 前端技术栈
- **HTML5/CSS3/JavaScript**
- **D3.js**：数据驱动的可视化库
- **Bootstrap**：响应式UI框架

## 项目结构

```
NetView/
├── docs/                    # 项目文档
│   ├── README.md           # 项目概述
│   ├── 01-project-setup.md # 项目环境搭建
│   ├── 02-backend-core.md  # 后端核心模块开发
│   ├── 03-api-server.md    # API服务开发
│   ├── 04-frontend-ui.md   # 前端界面开发
│   ├── 05-visualization.md # 可视化引擎开发
│   ├── 06-integration.md   # 系统集成测试
│   └── 07-deployment.md    # 部署和优化
├── backend/                # 后端代码
│   ├── model_parser.py     # 模型解析器
│   ├── graph_generator.py  # 图形数据生成器
│   ├── api_server.py       # API服务器
│   └── utils.py            # 工具函数
├── frontend/               # 前端代码
│   ├── index.html          # 主页面
│   ├── js/                 # JavaScript文件
│   └── css/                # 样式文件
├── tests/                  # 测试代码
├── examples/               # 示例模型
└── requirements.txt        # Python依赖
```

## 开发阶段

本项目按照以下7个阶段进行开发：

1. **项目环境搭建** - 创建项目结构，配置开发环境
2. **后端核心模块开发** - 实现模型解析和图形数据生成
3. **API服务开发** - 构建RESTful API接口
4. **前端界面开发** - 创建用户界面和基础交互
5. **可视化引擎开发** - 实现交互式图形可视化
6. **系统集成测试** - 整合各模块并进行测试
7. **部署和优化** - 部署配置和性能优化

## 快速开始

1. 克隆项目到本地
2. 安装Python依赖：`pip install -r requirements.txt`
3. 启动后端服务：`python backend/api_server.py`
4. 在浏览器中打开 `frontend/index.html`
5. 输入PyTorch模型代码，点击生成可视化图形

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。详细的贡献指南请参考各个开发阶段的文档。

## 许可证

MIT License
