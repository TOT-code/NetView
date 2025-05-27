#!/bin/bash

echo "================================================"
echo "NetView项目 Git仓库初始化脚本"
echo "================================================"
echo

# 检查Git是否安装
if ! command -v git &> /dev/null; then
    echo "错误: Git未安装，请先安装Git"
    exit 1
fi

# 检查是否已经是Git仓库
if [ -d ".git" ]; then
    echo "检测到已存在Git仓库"
    echo
else
    echo "正在初始化Git仓库..."
    git init
    if [ $? -ne 0 ]; then
        echo "错误: Git初始化失败"
        exit 1
    fi
    echo "Git仓库初始化成功！"
    echo
fi

# 主菜单函数
show_menu() {
    echo "请选择操作："
    echo "1. 配置Git用户信息"
    echo "2. 添加远程仓库"
    echo "3. 执行首次提交"
    echo "4. 查看Git状态"
    echo "5. 退出"
    echo
}

# 配置用户信息
config_user() {
    echo
    echo "配置Git用户信息"
    echo "================"
    read -p "请输入您的姓名: " username
    read -p "请输入您的邮箱: " email
    
    git config --global user.name "$username"
    git config --global user.email "$email"
    
    echo "用户信息配置完成！"
    echo "姓名: $username"
    echo "邮箱: $email"
    echo
    read -p "按回车键继续..."
}

# 添加远程仓库
add_remote() {
    echo
    echo "添加远程仓库"
    echo "============"
    echo "请先在GitHub/GitLab上创建仓库，然后复制仓库地址"
    echo
    read -p "请输入远程仓库地址: " remote_url
    
    git remote add origin "$remote_url"
    if [ $? -ne 0 ]; then
        echo "错误: 添加远程仓库失败"
    else
        echo "远程仓库添加成功！"
        echo "仓库地址: $remote_url"
    fi
    echo
    read -p "按回车键继续..."
}

# 首次提交
first_commit() {
    echo
    echo "执行首次提交"
    echo "============"
    echo "正在添加所有文件到暂存区..."
    git add .
    
    echo "正在创建首次提交..."
    git commit -m "配置: 初始化NetView项目环境和基础框架"
    
    if [ $? -ne 0 ]; then
        echo "错误: 提交失败"
        read -p "按回车键继续..."
        return
    fi
    
    echo "首次提交成功！"
    echo
    
    # 检查是否有远程仓库
    if git remote -v &> /dev/null && [ $(git remote | wc -l) -gt 0 ]; then
        read -p "检测到远程仓库，是否推送到远程？ (y/n): " push_choice
        if [[ $push_choice =~ ^[Yy]$ ]]; then
            echo "正在推送到远程仓库..."
            git push -u origin main
            if [ $? -ne 0 ]; then
                echo "推送失败，可能需要先创建远程仓库或检查网络连接"
            else
                echo "推送成功！"
            fi
        fi
    fi
    echo
    read -p "按回车键继续..."
}

# 查看Git状态
git_status() {
    echo
    echo "Git仓库状态"
    echo "==========="
    git status
    echo
    echo "最近的提交记录:"
    git log --oneline -5 2>/dev/null || echo "暂无提交记录"
    echo
    read -p "按回车键继续..."
}

# 主循环
while true; do
    show_menu
    read -p "请输入选择 (1-5): " choice
    
    case $choice in
        1)
            config_user
            ;;
        2)
            add_remote
            ;;
        3)
            first_commit
            ;;
        4)
            git_status
            ;;
        5)
            echo
            echo "Git仓库初始化完成！"
            echo
            echo "常用命令提醒："
            echo "  git status          - 查看状态"
            echo "  git add .           - 添加所有文件"
            echo "  git commit -m \"...\" - 提交更改"
            echo "  git push            - 推送到远程"
            echo
            echo "详细使用说明请查看: docs/git-guide.md"
            echo
            exit 0
            ;;
        *)
            echo "无效选择，请重新输入"
            echo
            ;;
    esac
done
