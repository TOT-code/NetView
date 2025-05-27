# NetView项目 Git仓库初始化脚本 (PowerShell版本)
# 编码: UTF-8 BOM

# 设置PowerShell编码
$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8

# 设置控制台标题
$host.UI.RawUI.WindowTitle = "NetView Git 初始化向导"

# 颜色配置
$Colors = @{
    Header = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "White"
    Menu = "DarkCyan"
}

# 显示标题
function Show-Header {
    Clear-Host
    Write-Host "================================================" -ForegroundColor $Colors.Header
    Write-Host "NetView项目 Git仓库初始化脚本" -ForegroundColor $Colors.Header
    Write-Host "================================================" -ForegroundColor $Colors.Header
    Write-Host ""
}

# 检查Git是否安装
function Test-GitInstallation {
    try {
        $gitVersion = git --version 2>$null
        if ($gitVersion) {
            Write-Host "✓ Git已安装: $gitVersion" -ForegroundColor $Colors.Success
            return $true
        }
    }
    catch {
        Write-Host "✗ 错误: Git未安装，请先安装Git" -ForegroundColor $Colors.Error
        Write-Host "下载地址: https://git-scm.com/download/win" -ForegroundColor $Colors.Info
        return $false
    }
}

# 检查是否为Git仓库
function Test-GitRepository {
    if (Test-Path ".git") {
        Write-Host "✓ 检测到已存在Git仓库" -ForegroundColor $Colors.Success
        return $true
    }
    else {
        Write-Host "○ 尚未初始化Git仓库" -ForegroundColor $Colors.Warning
        return $false
    }
}

# 初始化Git仓库
function Initialize-GitRepository {
    Write-Host "正在初始化Git仓库..." -ForegroundColor $Colors.Info
    try {
        git init
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Git仓库初始化成功！" -ForegroundColor $Colors.Success
            return $true
        }
        else {
            Write-Host "✗ Git初始化失败" -ForegroundColor $Colors.Error
            return $false
        }
    }
    catch {
        Write-Host "✗ Git初始化失败: $($_.Exception.Message)" -ForegroundColor $Colors.Error
        return $false
    }
}

# 显示主菜单
function Show-Menu {
    Write-Host ""
    Write-Host "请选择操作：" -ForegroundColor $Colors.Menu
    Write-Host "1. 配置Git用户信息" -ForegroundColor $Colors.Info
    Write-Host "2. 添加远程仓库" -ForegroundColor $Colors.Info
    Write-Host "3. 执行首次提交" -ForegroundColor $Colors.Info
    Write-Host "4. 查看Git状态" -ForegroundColor $Colors.Info
    Write-Host "5. 查看提交历史" -ForegroundColor $Colors.Info
    Write-Host "6. 推送到远程仓库" -ForegroundColor $Colors.Info
    Write-Host "7. 退出" -ForegroundColor $Colors.Info
    Write-Host ""
}

# 配置Git用户信息
function Set-GitUserInfo {
    Write-Host ""
    Write-Host "配置Git用户信息" -ForegroundColor $Colors.Header
    Write-Host "================" -ForegroundColor $Colors.Header
    
    # 显示当前配置
    $currentName = git config --global user.name 2>$null
    $currentEmail = git config --global user.email 2>$null
    
    if ($currentName) {
        Write-Host "当前姓名: $currentName" -ForegroundColor $Colors.Info
    }
    if ($currentEmail) {
        Write-Host "当前邮箱: $currentEmail" -ForegroundColor $Colors.Info
    }
    Write-Host ""
    
    $username = Read-Host "请输入您的姓名"
    if ([string]::IsNullOrWhiteSpace($username)) {
        Write-Host "✗ 姓名不能为空" -ForegroundColor $Colors.Error
        Read-Host "按回车键继续..."
        return
    }
    
    $email = Read-Host "请输入您的邮箱"
    if ([string]::IsNullOrWhiteSpace($email)) {
        Write-Host "✗ 邮箱不能为空" -ForegroundColor $Colors.Error
        Read-Host "按回车键继续..."
        return
    }
    
    try {
        git config --global user.name $username
        git config --global user.email $email
        
        Write-Host ""
        Write-Host "✓ 用户信息配置完成！" -ForegroundColor $Colors.Success
        Write-Host "姓名: $username" -ForegroundColor $Colors.Info
        Write-Host "邮箱: $email" -ForegroundColor $Colors.Info
    }
    catch {
        Write-Host "✗ 配置失败: $($_.Exception.Message)" -ForegroundColor $Colors.Error
    }
    
    Write-Host ""
    Read-Host "按回车键继续..."
}

# 添加远程仓库
function Add-RemoteRepository {
    Write-Host ""
    Write-Host "添加远程仓库" -ForegroundColor $Colors.Header
    Write-Host "============" -ForegroundColor $Colors.Header
    
    # 检查现有远程仓库
    $existingRemotes = git remote -v 2>$null
    if ($existingRemotes) {
        Write-Host "当前远程仓库:" -ForegroundColor $Colors.Info
        Write-Host $existingRemotes -ForegroundColor $Colors.Warning
        Write-Host ""
        
        $replace = Read-Host "已存在远程仓库，是否替换？(y/n)"
        if ($replace -eq 'y' -or $replace -eq 'Y') {
            git remote remove origin 2>$null
            Write-Host "✓ 已移除现有远程仓库" -ForegroundColor $Colors.Success
        }
        else {
            Write-Host "操作已取消" -ForegroundColor $Colors.Warning
            Read-Host "按回车键继续..."
            return
        }
    }
    
    Write-Host "请先在GitHub/GitLab上创建仓库，然后复制仓库地址" -ForegroundColor $Colors.Info
    Write-Host "示例: https://github.com/您的用户名/NetView.git" -ForegroundColor $Colors.Info
    Write-Host ""
    
    $remoteUrl = Read-Host "请输入远程仓库地址"
    if ([string]::IsNullOrWhiteSpace($remoteUrl)) {
        Write-Host "✗ 仓库地址不能为空" -ForegroundColor $Colors.Error
        Read-Host "按回车键继续..."
        return
    }
    
    try {
        git remote add origin $remoteUrl
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ 远程仓库添加成功！" -ForegroundColor $Colors.Success
            Write-Host "仓库地址: $remoteUrl" -ForegroundColor $Colors.Info
        }
        else {
            Write-Host "✗ 添加远程仓库失败" -ForegroundColor $Colors.Error
        }
    }
    catch {
        Write-Host "✗ 添加失败: $($_.Exception.Message)" -ForegroundColor $Colors.Error
    }
    
    Write-Host ""
    Read-Host "按回车键继续..."
}

# 执行首次提交
function Invoke-FirstCommit {
    Write-Host ""
    Write-Host "执行首次提交" -ForegroundColor $Colors.Header
    Write-Host "============" -ForegroundColor $Colors.Header
    
    # 检查是否有未提交的更改
    $status = git status --porcelain 2>$null
    if (-not $status) {
        Write-Host "✗ 没有需要提交的更改" -ForegroundColor $Colors.Warning
        Read-Host "按回车键继续..."
        return
    }
    
    Write-Host "检测到以下文件变化:" -ForegroundColor $Colors.Info
    git status --short
    Write-Host ""
    
    $confirm = Read-Host "是否提交这些更改？(y/n)"
    if ($confirm -ne 'y' -and $confirm -ne 'Y') {
        Write-Host "操作已取消" -ForegroundColor $Colors.Warning
        Read-Host "按回车键继续..."
        return
    }
    
    try {
        Write-Host "正在添加所有文件到暂存区..." -ForegroundColor $Colors.Info
        git add .
        
        Write-Host "正在创建首次提交..." -ForegroundColor $Colors.Info
        git commit -m "配置: 初始化NetView项目环境和基础框架"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ 首次提交成功！" -ForegroundColor $Colors.Success
            
            # 检查是否有远程仓库
            $remotes = git remote 2>$null
            if ($remotes) {
                Write-Host ""
                $push = Read-Host "检测到远程仓库，是否推送到远程？(y/n)"
                if ($push -eq 'y' -or $push -eq 'Y') {
                    Write-Host "正在推送到远程仓库..." -ForegroundColor $Colors.Info
                    
                    # 检查当前分支名
                    $branch = git branch --show-current 2>$null
                    if (-not $branch) {
                        $branch = "main"
                        git branch -M main
                    }
                    
                    git push -u origin $branch
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host "✓ 推送成功！" -ForegroundColor $Colors.Success
                    }
                    else {
                        Write-Host "✗ 推送失败，可能需要先创建远程仓库或检查权限" -ForegroundColor $Colors.Error
                    }
                }
            }
        }
        else {
            Write-Host "✗ 提交失败" -ForegroundColor $Colors.Error
        }
    }
    catch {
        Write-Host "✗ 提交失败: $($_.Exception.Message)" -ForegroundColor $Colors.Error
    }
    
    Write-Host ""
    Read-Host "按回车键继续..."
}

# 查看Git状态
function Show-GitStatus {
    Write-Host ""
    Write-Host "Git仓库状态" -ForegroundColor $Colors.Header
    Write-Host "===========" -ForegroundColor $Colors.Header
    
    try {
        Write-Host "工作区状态:" -ForegroundColor $Colors.Info
        git status
        
        Write-Host ""
        Write-Host "远程仓库:" -ForegroundColor $Colors.Info
        $remotes = git remote -v 2>$null
        if ($remotes) {
            Write-Host $remotes -ForegroundColor $Colors.Info
        }
        else {
            Write-Host "无远程仓库" -ForegroundColor $Colors.Warning
        }
        
        Write-Host ""
        Write-Host "当前分支:" -ForegroundColor $Colors.Info
        $branch = git branch --show-current 2>$null
        if ($branch) {
            Write-Host $branch -ForegroundColor $Colors.Info
        }
        else {
            Write-Host "无活动分支" -ForegroundColor $Colors.Warning
        }
    }
    catch {
        Write-Host "✗ 获取状态失败: $($_.Exception.Message)" -ForegroundColor $Colors.Error
    }
    
    Write-Host ""
    Read-Host "按回车键继续..."
}

# 查看提交历史
function Show-CommitHistory {
    Write-Host ""
    Write-Host "最近的提交记录" -ForegroundColor $Colors.Header
    Write-Host "==============" -ForegroundColor $Colors.Header
    
    try {
        $commits = git log --oneline -10 2>$null
        if ($commits) {
            Write-Host $commits -ForegroundColor $Colors.Info
        }
        else {
            Write-Host "暂无提交记录" -ForegroundColor $Colors.Warning
        }
    }
    catch {
        Write-Host "✗ 获取提交历史失败: $($_.Exception.Message)" -ForegroundColor $Colors.Error
    }
    
    Write-Host ""
    Read-Host "按回车键继续..."
}

# 推送到远程仓库
function Push-ToRemote {
    Write-Host ""
    Write-Host "推送到远程仓库" -ForegroundColor $Colors.Header
    Write-Host "==============" -ForegroundColor $Colors.Header
    
    # 检查是否有远程仓库
    $remotes = git remote 2>$null
    if (-not $remotes) {
        Write-Host "✗ 未配置远程仓库，请先添加远程仓库" -ForegroundColor $Colors.Error
        Read-Host "按回车键继续..."
        return
    }
    
    # 检查是否有提交
    $commits = git log --oneline -1 2>$null
    if (-not $commits) {
        Write-Host "✗ 没有可推送的提交，请先创建提交" -ForegroundColor $Colors.Error
        Read-Host "按回车键继续..."
        return
    }
    
    try {
        $branch = git branch --show-current 2>$null
        if (-not $branch) {
            $branch = "main"
            git branch -M main
        }
        
        Write-Host "正在推送分支 '$branch' 到远程仓库..." -ForegroundColor $Colors.Info
        git push -u origin $branch
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ 推送成功！" -ForegroundColor $Colors.Success
        }
        else {
            Write-Host "✗ 推送失败" -ForegroundColor $Colors.Error
            Write-Host "可能的原因:" -ForegroundColor $Colors.Warning
            Write-Host "- 远程仓库不存在" -ForegroundColor $Colors.Warning
            Write-Host "- 没有推送权限" -ForegroundColor $Colors.Warning
            Write-Host "- 网络连接问题" -ForegroundColor $Colors.Warning
        }
    }
    catch {
        Write-Host "✗ 推送失败: $($_.Exception.Message)" -ForegroundColor $Colors.Error
    }
    
    Write-Host ""
    Read-Host "按回车键继续..."
}

# 主程序
function Main {
    # 显示标题
    Show-Header
    
    # 检查Git安装
    if (-not (Test-GitInstallation)) {
        Read-Host "按回车键退出..."
        exit 1
    }
    
    # 检查并初始化Git仓库
    if (-not (Test-GitRepository)) {
        Write-Host ""
        $init = Read-Host "是否初始化Git仓库？(y/n)"
        if ($init -eq 'y' -or $init -eq 'Y') {
            if (-not (Initialize-GitRepository)) {
                Read-Host "按回车键退出..."
                exit 1
            }
        }
    }
    
    # 主循环
    do {
        Show-Header
        
        # 显示基本信息
        $currentBranch = git branch --show-current 2>$null
        $remoteInfo = git remote -v 2>$null
        
        Write-Host "当前状态:" -ForegroundColor $Colors.Info
        if ($currentBranch) {
            Write-Host "  分支: $currentBranch" -ForegroundColor $Colors.Success
        }
        if ($remoteInfo) {
            Write-Host "  远程: 已配置" -ForegroundColor $Colors.Success
        }
        else {
            Write-Host "  远程: 未配置" -ForegroundColor $Colors.Warning
        }
        
        Show-Menu
        $choice = Read-Host "请输入选择 (1-7)"
        
        switch ($choice) {
            "1" { Set-GitUserInfo }
            "2" { Add-RemoteRepository }
            "3" { Invoke-FirstCommit }
            "4" { Show-GitStatus }
            "5" { Show-CommitHistory }
            "6" { Push-ToRemote }
            "7" { 
                Write-Host ""
                Write-Host "Git仓库初始化完成！" -ForegroundColor $Colors.Success
                Write-Host ""
                Write-Host "常用命令提醒：" -ForegroundColor $Colors.Info
                Write-Host "  git status          - 查看状态" -ForegroundColor $Colors.Info
                Write-Host "  git add .           - 添加所有文件" -ForegroundColor $Colors.Info
                Write-Host "  git commit -m '...' - 提交更改" -ForegroundColor $Colors.Info
                Write-Host "  git push            - 推送到远程" -ForegroundColor $Colors.Info
                Write-Host ""
                Write-Host "详细使用说明请查看: docs/git-guide.md" -ForegroundColor $Colors.Info
                Write-Host ""
                exit 0
            }
            default { 
                Write-Host "无效选择，请重新输入" -ForegroundColor $Colors.Error
                Start-Sleep -Seconds 1
            }
        }
    } while ($true)
}

# 运行主程序
Main
