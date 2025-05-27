# PowerShell Git初始化脚本使用指南

## 🚀 快速开始

### 运行PowerShell脚本

1. **右键菜单运行**：
   - 在项目根目录找到 `init_git.ps1` 文件
   - 右键点击文件
   - 选择"使用PowerShell运行"

2. **PowerShell命令行运行**：
   ```powershell
   # 进入项目目录
   cd E:\DevCode\NetView
   
   # 运行脚本
   .\init_git.ps1
   ```

3. **如果遇到执行策略限制**：
   ```powershell
   # 临时允许执行PowerShell脚本
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   
   # 然后运行脚本
   .\init_git.ps1
   ```

## 📋 脚本功能

### 主要功能
1. **配置Git用户信息** - 设置全局用户名和邮箱
2. **添加远程仓库** - 连接到GitHub/GitLab仓库
3. **执行首次提交** - 提交项目文件到Git
4. **查看Git状态** - 检查仓库当前状态
5. **查看提交历史** - 显示最近的提交记录
6. **推送到远程仓库** - 同步到云端

### 特色功能
- ✅ **彩色界面** - 不同颜色显示不同状态
- ✅ **智能检测** - 自动检测Git安装和仓库状态
- ✅ **错误处理** - 友好的错误提示和解决建议
- ✅ **中文支持** - 完美支持中文显示
- ✅ **分支管理** - 自动处理main分支创建

## 🎯 使用流程

### 首次使用建议流程：

1. **运行脚本**
   ```powershell
   .\init_git.ps1
   ```

2. **配置用户信息** (选择1)
   - 输入您的姓名
   - 输入您的邮箱地址

3. **添加远程仓库** (选择2)
   - 先在GitHub创建新仓库
   - 复制仓库地址并输入

4. **执行首次提交** (选择3)
   - 确认要提交的文件
   - 选择是否推送到远程

## 🔧 故障排除

### 常见问题解决

**1. 执行策略错误**
```
无法加载文件 init_git.ps1，因为在此系统上禁止运行脚本
```
解决方法：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**2. Git未安装**
```
Git未安装，请先安装Git
```
解决方法：
- 访问 https://git-scm.com/download/win
- 下载并安装Git for Windows

**3. 推送失败**
```
推送失败，可能需要先创建远程仓库或检查权限
```
解决方法：
- 确保在GitHub上已创建仓库
- 检查仓库地址是否正确
- 确认有推送权限

**4. 中文乱码**
```
界面显示乱码字符
```
解决方法：
- PowerShell脚本已优化编码，应该不会出现乱码
- 如果仍有问题，尝试在PowerShell中运行：
  ```powershell
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  ```

## 📚 Git命令对照

脚本操作对应的Git命令：

| 脚本功能 | 对应Git命令 |
|---------|------------|
| 配置用户信息 | `git config --global user.name "姓名"` <br> `git config --global user.email "邮箱"` |
| 添加远程仓库 | `git remote add origin 仓库地址` |
| 首次提交 | `git add .` <br> `git commit -m "提交信息"` |
| 推送到远程 | `git push -u origin main` |
| 查看状态 | `git status` |
| 查看历史 | `git log --oneline` |

## 🎨 脚本界面

### 颜色说明
- 🔵 **蓝色** - 标题和重要信息
- 🟢 **绿色** - 成功操作和确认信息
- 🟡 **黄色** - 警告和注意事项
- 🔴 **红色** - 错误信息
- ⚪ **白色** - 一般信息和菜单

### 状态指示
- ✓ 成功完成
- ✗ 操作失败
- ○ 待处理状态

## 🔗 相关文档

- [Git使用指南](git-guide.md) - 详细的Git操作说明
- [项目README](../README.md) - 项目总体介绍
- [开发计划](development-plan.md) - 完整开发规划

## 💡 小贴士

1. **首次使用**：建议按照1→2→3的顺序操作
2. **日常使用**：通常只需要选择4查看状态，然后选择6推送
3. **安全操作**：提交前先查看状态，确认要提交的文件
4. **备份重要**：定期推送到远程仓库，避免代码丢失

---

**如果遇到其他问题，请参考 [Git使用指南](git-guide.md) 或查看Git官方文档。**
