@echo off
echo ================================================
echo NetView项目 Git仓库初始化脚本
echo ================================================
echo.

REM 检查是否已经是Git仓库
if exist .git (
    echo 检测到已存在Git仓库
    echo.
    goto :menu
)

echo 正在初始化Git仓库...
git init
if %errorlevel% neq 0 (
    echo 错误: Git初始化失败，请检查Git是否正确安装
    pause
    exit /b 1
)

echo Git仓库初始化成功！
echo.

:menu
echo 请选择操作：
echo 1. 配置Git用户信息
echo 2. 添加远程仓库
echo 3. 执行首次提交
echo 4. 查看Git状态
echo 5. 退出
echo.
set /p choice="请输入选择 (1-5): "

if "%choice%"=="1" goto :config_user
if "%choice%"=="2" goto :add_remote
if "%choice%"=="3" goto :first_commit
if "%choice%"=="4" goto :git_status
if "%choice%"=="5" goto :end
echo 无效选择，请重新输入
goto :menu

:config_user
echo.
echo 配置Git用户信息
echo ================
set /p username="请输入您的姓名: "
set /p email="请输入您的邮箱: "

git config --global user.name "%username%"
git config --global user.email "%email%"

echo 用户信息配置完成！
echo 姓名: %username%
echo 邮箱: %email%
echo.
pause
goto :menu

:add_remote
echo.
echo 添加远程仓库
echo ============
echo 请先在GitHub/GitLab上创建仓库，然后复制仓库地址
echo.
set /p remote_url="请输入远程仓库地址: "

git remote add origin %remote_url%
if %errorlevel% neq 0 (
    echo 错误: 添加远程仓库失败
) else (
    echo 远程仓库添加成功！
    echo 仓库地址: %remote_url%
)
echo.
pause
goto :menu

:first_commit
echo.
echo 执行首次提交
echo ============
echo 正在添加所有文件到暂存区...
git add .

echo 正在创建首次提交...
git commit -m "配置: 初始化NetView项目环境和基础框架"

if %errorlevel% neq 0 (
    echo 错误: 提交失败
    pause
    goto :menu
)

echo 首次提交成功！
echo.

REM 检查是否有远程仓库
git remote -v >nul 2>&1
if %errorlevel% equ 0 (
    echo 检测到远程仓库，是否推送到远程？
    set /p push_choice="推送到远程仓库？ (y/n): "
    if /i "%push_choice%"=="y" (
        echo 正在推送到远程仓库...
        git push -u origin main
        if %errorlevel% neq 0 (
            echo 推送失败，可能需要先创建远程仓库或检查网络连接
        ) else (
            echo 推送成功！
        )
    )
)
echo.
pause
goto :menu

:git_status
echo.
echo Git仓库状态
echo ===========
git status
echo.
echo 最近的提交记录:
git log --oneline -5
echo.
pause
goto :menu

:end
echo.
echo Git仓库初始化完成！
echo.
echo 常用命令提醒：
echo   git status          - 查看状态
echo   git add .           - 添加所有文件
echo   git commit -m "..." - 提交更改
echo   git push            - 推送到远程
echo.
echo 详细使用说明请查看: docs/git-guide.md
echo.
pause
