# Git网络连接问题解决指南

## 🔍 问题诊断

### 常见错误信息
```
fatal: unable to access 'https://github.com/ToT-code/NetView.git/': Failed to connect to github.com port 443: Timed out
```

这表示无法连接到GitHub的HTTPS端口（443）。

## 🛠️ 解决方案

### 方案1：检查网络连接
```bash
# 测试基本网络连接
ping github.com

# 测试HTTPS连接
curl -I https://github.com
```

### 方案2：使用代理（如果在公司网络）
```bash
# 设置HTTP代理
git config --global http.proxy http://proxy.company.com:port
git config --global https.proxy https://proxy.company.com:port

# 取消代理设置
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### 方案3：修改DNS设置
1. **使用公共DNS服务器**：
   - 8.8.8.8 (Google DNS)
   - 114.114.114.114 (国内DNS)
   - 223.5.5.5 (阿里DNS)

2. **刷新DNS缓存**：
   ```cmd
   # Windows
   ipconfig /flushdns
   
   # macOS
   sudo dscacheutil -flushcache
   
   # Linux
   sudo systemctl restart systemd-resolved
   ```

### 方案4：使用SSH代替HTTPS
```bash
# 移除HTTPS远程仓库
git remote remove origin

# 添加SSH远程仓库
git remote add origin git@github.com:ToT-code/NetView.git

# 配置SSH密钥（如果还没有）
ssh-keygen -t ed25519 -C "xjl2026@qq.com"
```

### 方案5：修改hosts文件（临时方案）
在 `C:\Windows\System32\drivers\etc\hosts` 文件中添加：
```
140.82.113.3 github.com
140.82.114.4 api.github.com
```

### 方案6：使用GitHub镜像站点
```bash
# 使用国内镜像
git remote set-url origin https://github.com.cnpmjs.org/ToT-code/NetView.git

# 或者使用FastGit镜像
git remote set-url origin https://hub.fastgit.xyz/ToT-code/NetView.git
```

## 🔧 PowerShell脚本修复版

如果您使用PowerShell脚本遇到网络问题，可以尝试以下方法：

### 1. 网络诊断功能
在PowerShell中运行：
```powershell
# 测试GitHub连接
Test-NetConnection github.com -Port 443

# 检查DNS解析
Resolve-DnsName github.com
```

### 2. 临时解决方案
```powershell
# 在PowerShell脚本中添加网络检查
function Test-GitHubConnection {
    try {
        $result = Test-NetConnection github.com -Port 443 -WarningAction SilentlyContinue
        return $result.TcpTestSucceeded
    }
    catch {
        return $false
    }
}
```

## 📋 分步排查流程

### 第1步：基本连接测试
```bash
ping github.com
```
- ✅ 如果能ping通，说明基本网络正常
- ❌ 如果ping不通，检查网络连接和DNS

### 第2步：端口连接测试
```bash
telnet github.com 443
```
- ✅ 如果能连接，说明HTTPS端口正常
- ❌ 如果连接失败，可能被防火墙阻止

### 第3步：Git配置检查
```bash
git config --list | grep -E "(http|https|proxy)"
```
检查是否有错误的代理配置

### 第4步：尝试不同的连接方式
```bash
# 尝试SSH
git ls-remote git@github.com:ToT-code/NetView.git

# 尝试HTTPS
git ls-remote https://github.com/ToT-code/NetView.git
```

## 🌐 替代方案

### 1. 本地开发优先
```bash
# 先在本地进行开发
git add .
git commit -m "本地开发进度保存"

# 网络恢复后再推送
git push origin main
```

### 2. 使用其他Git托管服务
- **Gitee** (码云): https://gitee.com
- **GitLab**: https://gitlab.com
- **Bitbucket**: https://bitbucket.org

### 3. 离线备份
```bash
# 创建项目压缩包备份
tar -czf netview-backup-$(date +%Y%m%d).tar.gz ./NetView
```

## 🔄 网络恢复后的操作

```bash
# 1. 测试连接
git remote -v

# 2. 拉取最新更改（如果有其他协作者）
git pull

# 3. 推送本地提交
git push

# 4. 验证同步
git status
```

## 💡 预防措施

1. **定期备份**：经常将代码推送到远程仓库
2. **多点备份**：使用多个Git托管服务
3. **本地备份**：定期创建项目备份
4. **网络工具**：配置稳定的网络环境

## 🆘 紧急情况处理

如果长时间无法连接GitHub：

1. **继续本地开发**：所有Git功能在本地仍然可用
2. **使用替代托管**：临时使用Gitee等国内服务
3. **离线协作**：通过文件共享方式协作
4. **寻求帮助**：联系网络管理员或IT支持

---

**记住**：Git是分布式版本控制系统，即使无法连接远程仓库，本地的版本控制功能依然完全可用！
