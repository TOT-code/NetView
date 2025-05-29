# NetView项目Git推送操作指南

## 📋 准备工作

在开始之前，请确保：
- 已安装Git
- 拥有GitHub账户访问权限
- SSH密钥已配置（使用SSH地址时）

## 🔍 第一步：检查当前Git状态

```bash
# 检查是否已经是git仓库
git status
```

**如果显示错误"not a git repository"，则需要初始化：**
```bash
git init
```

## ⚙️ 第二步：配置Git用户信息

```bash
# 设置用户名（替换为你的GitHub用户名）
git config --global user.name "TOT-code"

# 设置邮箱（替换为你的GitHub邮箱）
git config --global user.email "your-email@example.com"

# 验证配置
git config --global user.name
git config --global user.email
```

## 🔗 第三步：添加远程仓库

```bash
# 查看当前远程仓库配置
git remote -v

# 如果没有远程仓库，添加你的GitHub仓库
git remote add origin git@github.com:TOT-code/NetView.git

# 如果已存在远程仓库但地址不对，更新地址
git remote set-url origin git@github.com:TOT-code/NetView.git

# 验证远程仓库配置
git remote -v
```

## 📁 第四步：添加文件到暂存区

```bash
# 查看当前文件状态
git status

# 添加所有文件到暂存区
git add .

# 或者选择性添加文件
# git add README.md
# git add backend/
# git add frontend/
# git add config.py
# git add requirements-minimal.txt

# 验证暂存区状态
git status
```

## 💬 第五步：创建提交

```bash
# 创建首次提交
git commit -m "🎉 初始化NetView项目：PyTorch模型可视化工具

✨ 功能特性:
- FastAPI后端架构，支持模型解析和分析
- 前端可视化界面，基于Vis.js网络图
- 多维度模型分析（AST、运行时、动态图）
- 智能架构模式识别
- 多格式导出支持（JSON/PNG/SVG）
- 完整的API文档和配置系统

🏗️ 项目结构:
- backend/: FastAPI服务端
- frontend/: 前端界面
- examples/: 示例模型
- 配置文件和启动脚本

📝 已知问题:
- 前端显示效果需要优化
- 界面交互体验待改进"
```

## 🚀 第六步：推送到GitHub

```bash
# 查看分支信息
git branch

# 如果需要重命名主分支为main
git branch -M main

# 首次推送到远程仓库
git push -u origin master

# 后续推送可以直接使用
# git push
```

## 🔧 故障排除

### 如果推送失败

**1. SSH密钥问题**
```bash
# 测试SSH连接
ssh -T git@github.com
```

**2. 如果远程仓库已有内容**
```bash
# 拉取远程内容并合并
git pull origin main --allow-unrelated-histories

# 解决冲突后重新推送
git push -u origin main
```

**3. 使用HTTPS替代SSH**
```bash
# 更改为HTTPS地址
git remote set-url origin https://github.com/TOT-code/NetView.git

# 推送时会要求输入用户名和密码/token
git push -u origin main
```

## 📋 完整命令清单（复制粘贴执行）

```bash
# === 基础检查和配置 ===
git status
git config --global user.name "TOT-code"
git config --global user.email "your-email@example.com"

# === 如果不是git仓库，执行初始化 ===
# git init

# === 配置远程仓库 ===
git remote add origin git@github.com:TOT-code/NetView.git
# 或者如果已存在：git remote set-url origin git@github.com:TOT-code/NetView.git

# === 添加文件并提交 ===
git add .
git commit -m "🎉 初始化NetView项目：PyTorch模型可视化工具

✨ 功能特性:
- FastAPI后端架构，支持模型解析和分析
- 前端可视化界面，基于Vis.js网络图
- 多维度模型分析（AST、运行时、动态图）
- 智能架构模式识别
- 多格式导出支持（JSON/PNG/SVG）
- 完整的API文档和配置系统

🏗️ 项目结构:
- backend/: FastAPI服务端
- frontend/: 前端界面
- examples/: 示例模型
- 配置文件和启动脚本

📝 已知问题:
- 前端显示效果需要优化
- 界面交互体验待改进"

# === 推送到GitHub ===
git branch -M main
git push -u origin main
```

## 🎯 验证推送成功

推送完成后，访问你的GitHub仓库页面确认：
- 文件已上传成功
- 提交信息显示正确
- 所有目录结构完整

GitHub仓库地址：https://github.com/TOT-code/NetView

## 📝 后续操作建议

1. **在GitHub上完善项目信息**
   - 添加项目描述
   - 设置项目标签
   - 启用Issues和Wiki

2. **更新README.md**
   - 替换GitHub地址占位符
   - 添加项目截图
   - 完善使用说明

3. **设置GitHub Pages**（可选）
   - 展示项目前端界面
   - 提供在线演示

## ⚠️ 注意事项

1. **敏感信息检查**：确保没有提交敏感信息（API密钥、密码等）
2. **.gitignore文件**：项目已包含完整的.gitignore文件
3. **分支管理**：建议使用main分支作为主分支
4. **提交信息**：使用清晰的提交信息便于后续维护

---

💡 **提示**：如果在执行过程中遇到任何问题，可以查看具体的错误信息并根据错误提示进行相应处理。
