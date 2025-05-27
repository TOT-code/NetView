# SSH密钥配置完整指南

## 🔐 SSH密钥生成和配置

### 第1步：完成SSH密钥生成

如果SSH密钥生成过程还在进行中，按以下步骤操作：

```bash
# 如果提示输入文件保存位置，直接按回车使用默认位置
Enter file in which to save the key (/c/Users/你的用户名/.ssh/id_ed25519): [直接回车]

# 如果提示输入密码，可以设置密码或直接回车不设密码
Enter passphrase (empty for no passphrase): [回车或输入密码]
Enter same passphrase again: [回车或再次输入相同密码]
```

### 第2步：查看生成的SSH公钥

```bash
# 查看公钥内容
cat ~/.ssh/id_ed25519.pub

# 或者在Windows中使用
type %USERPROFILE%\.ssh\id_ed25519.pub

# 或者使用PowerShell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

### 第3步：复制SSH公钥到GitHub

1. **登录GitHub账户**
2. **进入设置页面**：
   - 点击右上角头像 → Settings
3. **添加SSH密钥**：
   - 左侧菜单选择 "SSH and GPG keys"
   - 点击 "New SSH key"
   - Title：输入描述，如 "NetView Development"
   - Key：粘贴刚才复制的公钥内容
   - 点击 "Add SSH key"

### 第4步：测试SSH连接

```bash
# 测试SSH连接到GitHub
ssh -T git@github.com

# 首次连接会提示，输入 yes 确认
The authenticity of host 'github.com' can't be established.
# 输入 yes 并回车
```

成功的话会看到：
```
Hi ToT-code! You've successfully authenticated, but GitHub does not provide shell access.
```

### 第5步：修改Git远程仓库URL

```bash
# 查看当前远程仓库
git remote -v

# 将HTTPS URL改为SSH URL
git remote set-url origin git@github.com:ToT-code/NetView.git

# 验证修改
git remote -v
```

## 🚀 完整的解决流程

### 一键修复脚本（PowerShell）

```powershell
# 1. 检查SSH密钥是否存在
if (Test-Path "$env:USERPROFILE\.ssh\id_ed25519.pub") {
    Write-Host "✓ SSH密钥已存在" -ForegroundColor Green
    Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
} else {
    Write-Host "⚠ SSH密钥不存在，请先生成" -ForegroundColor Yellow
}

# 2. 测试SSH连接
Write-Host "`n测试SSH连接..." -ForegroundColor Cyan
ssh -T git@github.com

# 3. 修改远程仓库URL
Write-Host "`n修改远程仓库URL..." -ForegroundColor Cyan
git remote set-url origin git@github.com:ToT-code/NetView.git
git remote -v

# 4. 测试推送
Write-Host "`n测试推送..." -ForegroundColor Cyan
git push -u origin main
```

## 📋 故障排除

### 问题1：Permission denied (publickey)
```bash
# 检查SSH代理
ssh-add -l

# 如果显示 "Could not open a connection to your authentication agent"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 问题2：SSH密钥未找到
```bash
# 重新生成SSH密钥
ssh-keygen -t ed25519 -C "your-email@example.com"
```

### 问题3：仍然无法连接
```bash
# 使用详细模式查看连接过程
ssh -vT git@github.com
```

## 🔄 验证完整流程

### 最终验证步骤：

1. **SSH连接测试**：
   ```bash
   ssh -T git@github.com
   ```

2. **推送测试**：
   ```bash
   git push origin main
   ```

3. **查看GitHub仓库**：
   确认代码已成功推送

## 💡 优势说明

### 使用SSH的好处：
- ✅ **绕过HTTPS端口限制**
- ✅ **更安全的认证方式**
- ✅ **无需每次输入密码**
- ✅ **适合企业网络环境**

### SSH vs HTTPS对比：
| 特性 | SSH | HTTPS |
|-----|-----|-------|
| 端口 | 22 | 443 |
| 认证 | 密钥 | 用户名密码/Token |
| 防火墙友好 | ✅ | ❌（可能被阻） |
| 设置复杂度 | 中等 | 简单 |

## 🎯 下一步操作

SSH配置完成后，您就可以：

1. **正常使用Git推送**：
   ```bash
   git add .
   git commit -m "配置: 完成SSH连接配置"
   git push origin main
   ```

2. **使用PowerShell脚本**：
   ```powershell
   .\init_git.ps1
   ```

3. **继续项目开发**：
   所有Git操作都会通过SSH正常工作

---

**重要提醒**：SSH密钥是您的私人认证凭据，请妥善保管私钥文件，不要分享给他人！
