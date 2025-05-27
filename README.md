# NetView - PyTorch网络结构可视化工具

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.9+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

NetView是一个专为PyTorch深度学习模型设计的网络结构可视化工具，能够解析PyTorch模型定义代码并生成交互式的网络结构图，帮助开发者更好地理解和分析深度学习模型的架构。

## ✨ 主要功能

- 🔍 **自动解析PyTorch模型**：输入模型定义代码，自动提取网络结构信息
- 📊 **详细信息展示**：显示层类型、名称、参数数量和tensor形状变化
- 🖱️ **交互式可视化**：支持拖拽、缩放、点击查看详情等交互操作
- 💾 **静态图片导出**：支持导出为PNG、SVG等格式的静态图片
- 🌐 **Web界面**：基于Web的用户友好界面，无需安装额外软件

## 🏗️ 项目结构

```
NetView/
├── .gitignore             # Git忽略文件配置
├── config.py              # 项目配置文件
├── requirements.txt        # Python依赖包
├── README.md              # 项目说明文档
├── backend/               # 后端Python代码
│   ├── __init__.py
│   └── modules/          # 核心模块目录
├── frontend/             # 前端Web代码
│   ├── index.html        # 主页面
│   ├── css/             # 样式文件
│   │   └── styles.css   # 主样式文件
│   └── js/              # JavaScript文件
│       ├── api_client.js      # API客户端
│       ├── visualization.js   # 可视化引擎
│       └── ui_handler.js      # UI交互处理
├── tests/               # 测试代码
│   ├── unit/           # 单元测试
│   └── integration/    # 集成测试
├── examples/            # 示例模型
│   └── simple_cnn.py   # 简单CNN示例
└── docs/               # 项目文档
    ├── README.md                # 项目概述
    ├── git-guide.md            # Git使用指南
    ├── development-plan.md      # 开发计划总览
    ├── 01-project-setup.md      # 第一阶段：项目环境搭建
    ├── 02-backend-core.md       # 第二阶段：后端核心模块开发
    ├── 03-api-server.md         # 第三阶段：API服务开发
    ├── 04-frontend-ui.md        # 第四阶段：前端界面开发
    ├── 05-visualization.md      # 第五阶段：可视化引擎开发
    ├── 06-integration.md        # 第六阶段：系统集成测试
    └── 07-deployment.md         # 第七阶段：部署和优化
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- 现代Web浏览器

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/YOUR_USERNAME/netview.git
cd netview
```

2. **创建虚拟环境**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **查看前端界面**
```bash
# 进入前端目录
cd frontend

# 启动简单HTTP服务器
python -m http.server 8080

# 在浏览器中访问
# http://localhost:8080
```

### 使用示例

1. 在浏览器中打开 `http://localhost:8080`
2. 在左侧代码输入框中输入PyTorch模型代码
3. 点击"解析模型"按钮
4. 在右侧查看生成的网络结构图
5. 使用鼠标拖拽节点调整布局
6. 点击导出按钮保存图片

## 📖 详细文档

- 📚 [开发计划总览](docs/development-plan.md) - 完整的开发规划和时间安排
- 🔧 [Git使用指南](docs/git-guide.md) - 项目Git管理和协作指南
- 🏗️ [项目搭建指南](docs/01-project-setup.md) - 开发环境配置
- 💻 [后端开发指南](docs/02-backend-core.md) - 核心模块开发
- 🌐 [API接口文档](docs/03-api-server.md) - API服务设计
- 🎨 [前端开发指南](docs/04-frontend-ui.md) - 界面开发
- 📊 [可视化开发](docs/05-visualization.md) - 图形渲染引擎
- 🧪 [测试和集成](docs/06-integration.md) - 系统测试
- 🚀 [部署指南](docs/07-deployment.md) - 生产环境部署

## 🛠️ 开发计划

项目采用7个阶段的渐进式开发方式：

### ✅ 第一阶段：项目环境搭建（已完成）
- [x] 项目目录结构创建
- [x] 开发环境配置
- [x] 前端基础框架
- [x] Git版本控制设置

### 🔄 第二阶段：后端核心模块开发（进行中）
- [ ] PyTorch模型解析器
- [ ] 网络结构分析算法
- [ ] 图形数据生成器

### 📅 后续阶段
- 第三阶段：API服务开发
- 第四阶段：前端界面开发
- 第五阶段：可视化引擎开发
- 第六阶段：系统集成测试
- 第七阶段：部署和优化

**预计总开发时间**：5-7周

## 🔧 技术栈

### 后端技术
- **Python 3.8+**：主要开发语言
- **PyTorch**：模型解析和分析
- **Flask/FastAPI**：Web API框架
- **NetworkX**：图形数据处理

### 前端技术
- **HTML5/CSS3/JavaScript**：基础Web技术
- **D3.js**：数据可视化
- **Bootstrap**：UI框架
- **现代浏览器API**：交互和导出功能

### 开发工具
- **Git**：版本控制
- **Python虚拟环境**：依赖管理
- **VS Code**：推荐开发环境

## 📋 Git管理

本项目使用Git进行版本控制，具体使用方法请参考：

### 快速开始Git
```bash
# 初始化Git仓库
git init

# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/netview.git

# 首次提交
git add .
git commit -m "初始化NetView项目"
git push -u origin main
```

### 日常开发流程
```bash
# 查看状态
git status

# 添加更改
git add .

# 提交更改
git commit -m "功能: 添加新功能描述"

# 推送到远程
git push
```

更多详细的Git使用说明，请查看 [Git使用指南](docs/git-guide.md)。

## 🎯 功能特性

### 当前已实现
- ✅ 响应式Web界面
- ✅ 代码输入和编辑
- ✅ 基础UI组件和交互
- ✅ 可视化框架搭建
- ✅ 项目配置管理

### 开发中
- 🔄 PyTorch模型解析
- 🔄 网络结构分析
- 🔄 图形数据生成

### 计划中
- 📋 交互式可视化
- 📋 多种导出格式
- 📋 示例模型库
- 📋 性能优化

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/新功能`)
3. 提交更改 (`git commit -m '功能: 添加新功能'`)
4. 推送到分支 (`git push origin feature/新功能`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目地址：[https://github.com/YOUR_USERNAME/netview](https://github.com/YOUR_USERNAME/netview)
- 问题反馈：[Issues](https://github.com/YOUR_USERNAME/netview/issues)
- 文档：[项目文档](docs/)

## 🙏 致谢

感谢以下开源项目：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [D3.js](https://d3js.org/) - 数据可视化库
- [Bootstrap](https://getbootstrap.com/) - UI框架
- [Font Awesome](https://fontawesome.com/) - 图标库

---

**⭐ 如果这个项目对您有帮助，请给它一个Star！**
