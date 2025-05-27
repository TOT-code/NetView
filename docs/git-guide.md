# NetView项目 Git使用指南

## 📋 目录
- [快速开始](#快速开始)
- [日常操作](#日常操作)
- [分支管理](#分支管理)
- [版本标签](#版本标签)
- [远程仓库](#远程仓库)
- [常用命令速查](#常用命令速查)
- [问题解决](#问题解决)

## 🚀 快速开始

### 1. 初始化Git仓库

如果是全新项目：
```bash
# 进入项目目录
cd E:\DevCode\NetView

# 初始化Git仓库
git init

# 配置用户信息（首次使用需要设置）
git config --global user.name "您的姓名"
git config --global user.email "您的邮箱@example.com"

# 添加所有文件到暂存区
git add .

# 创建首次提交
git commit -m "初始化NetView项目"
```

### 2. 连接远程仓库

```bash
# 添加远程仓库（将YOUR_USERNAME和YOUR_REPO替换为实际值）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 或者使用SSH（推荐，需要先配置SSH密钥）
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git

# 推送到远程仓库
git push -u origin main
```

### 3. SSH密钥配置（推荐）

```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "您的邮箱@example.com"

# 查看公钥内容
cat ~/.ssh/id_ed25519.pub

# 将公钥内容复制到GitHub/GitLab的SSH Keys设置中
```

## 📝 日常操作

### 基本工作流程

```bash
# 1. 查看当前状态
git status

# 2. 查看文件差异
git diff

# 3. 添加文件到暂存区
git add .                    # 添加所有文件
git add filename.py          # 添加特定文件
git add backend/             # 添加特定目录

# 4. 提交更改
git commit -m "功能: 添加模型解析器"

# 5. 推送到远程仓库
git push
```

### 提交信息规范

建议使用以下格式：
```
类型: 简短描述

详细说明（可选）
```

**类型标识：**
- `功能:` - 新功能
- `修复:` - Bug修复
- `文档:` - 文档更新
- `样式:` - 代码格式调整
- `重构:` - 代码重构
- `测试:` - 测试相关
- `配置:` - 配置文件更改

**示例：**
```bash
git commit -m "功能: 实现PyTorch模型解析器"
git commit -m "修复: 解决可视化渲染问题"
git commit -m "文档: 更新README和API文档"
```

## 🌿 分支管理

### 基本分支操作

```bash
# 查看所有分支
git branch -a

# 创建新分支
git branch feature/model-parser

# 切换分支
git checkout feature/model-parser

# 创建并切换到新分支（快捷方式）
git checkout -b feature/visualization

# 合并分支到main
git checkout main
git merge feature/model-parser

# 删除已合并的分支
git branch -d feature/model-parser
```

### 推荐的分支策略

对于个人项目，建议使用简单的分支策略：

- **main分支**：稳定的生产版本
- **feature/功能名**：开发新功能时使用
- **fix/问题描述**：修复Bug时使用

## 🏷️ 版本标签

```bash
# 创建轻量标签
git tag v0.1.0

# 创建带注释的标签（推荐）
git tag -a v0.1.0 -m "NetView首个版本发布"

# 查看所有标签
git tag

# 推送标签到远程
git push origin v0.1.0

# 推送所有标签
git push origin --tags

# 检出特定标签
git checkout v0.1.0
```

## 🌐 远程仓库

### 常用远程操作

```bash
# 查看远程仓库
git remote -v

# 从远程仓库拉取更新
git pull

# 推送到远程仓库
git push

# 推送新分支到远程
git push -u origin feature/new-feature

# 删除远程分支
git push origin --delete feature/old-feature
```

### 克隆项目（其他设备）

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/netview.git

# 进入项目目录
cd netview

# 安装依赖
pip install -r requirements.txt

# 查看项目状态
git status
```

## ⚡ 常用命令速查

### 查看和比较
```bash
git status              # 查看工作区状态
git log                 # 查看提交历史
git log --oneline       # 简洁的提交历史
git diff                # 查看工作区变化
git diff --staged       # 查看暂存区变化
git show COMMIT_ID      # 查看特定提交的详情
```

### 撤销和回滚
```bash
git checkout filename   # 撤销工作区文件的修改
git reset HEAD filename # 从暂存区移除文件
git reset --soft HEAD~1 # 撤销最后一次提交，保留更改
git reset --hard HEAD~1 # 撤销最后一次提交，丢弃更改
git revert COMMIT_ID    # 创建新提交来撤销指定提交
```

### 暂存和恢复
```bash
git stash               # 暂存当前工作区更改
git stash pop          # 恢复最近的暂存
git stash list         # 查看所有暂存
git stash drop         # 删除最近的暂存
```

## 🔧 问题解决

### 常见问题及解决方案

**1. 提交信息写错了**
```bash
# 修改最后一次提交信息
git commit --amend -m "新的提交信息"
```

**2. 忘记添加文件到最后一次提交**
```bash
git add forgotten_file.py
git commit --amend --no-edit
```

**3. 推送被拒绝（远程有新提交）**
```bash
git pull --rebase
git push
```

**4. 误删文件恢复**
```bash
git checkout HEAD -- filename
```

**5. 查看文件修改历史**
```bash
git log --follow -p filename
```

**6. 回到特定版本**
```bash
# 临时回到某个版本查看
git checkout COMMIT_ID

# 回到main分支最新版本
git checkout main
```

## 📅 开发工作流建议

### 日常开发流程

1. **开始新功能**
   ```bash
   git checkout main
   git pull
   git checkout -b feature/功能名称
   ```

2. **开发过程中**
   ```bash
   # 频繁提交，记录开发进度
   git add .
   git commit -m "功能: 完成XX模块基础框架"
   ```

3. **功能完成后**
   ```bash
   # 切换到main分支
   git checkout main
   git pull
   
   # 合并功能分支
   git merge feature/功能名称
   
   # 推送到远程
   git push
   
   # 删除功能分支
   git branch -d feature/功能名称
   ```

4. **发布版本**
   ```bash
   git tag -a v0.2.0 -m "版本0.2.0: 新增可视化功能"
   git push origin v0.2.0
   ```

## 🛡️ 最佳实践

1. **频繁提交**：每完成一个小功能就提交一次
2. **有意义的提交信息**：让其他人（包括未来的自己）能快速理解更改
3. **定期推送**：避免本地代码丢失
4. **使用分支**：为新功能创建分支，保持main分支稳定
5. **定期拉取**：保持与远程仓库同步
6. **检查.gitignore**：确保不提交不必要的文件

## 📚 更多资源

- [Git官方文档](https://git-scm.com/doc)
- [GitHub使用指南](https://guides.github.com/)
- [Git可视化学习](https://learngitbranching.js.org/)

---

**注意**: 这份指南针对个人项目开发场景编写。如果项目发展为团队协作，建议采用更完善的Git Flow或GitHub Flow工作流。
