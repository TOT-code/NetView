#!/usr/bin/env python3
"""
NetView 简单启动脚本
"""

import os
import sys
import time
import socket
import subprocess
from pathlib import Path

def check_port(host, port):
    """检查端口是否被占用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0  # 返回True表示端口可用
    except:
        return False

def find_available_port(start_port, host='localhost'):
    """查找可用端口"""
    port = start_port
    while port < start_port + 100:  # 最多尝试100个端口
        if check_port(host, port):
            return port
        port += 1
    return None

def main():
    # 获取项目根目录
    project_root = Path(__file__).parent
    
    print("🌟 NetView 简单启动器")
    print("=" * 40)
    
    # 检查基本文件
    backend_file = project_root / "backend" / "api" / "main.py"
    frontend_file = project_root / "frontend" / "index.html"
    
    if not backend_file.exists():
        print(f"❌ 后端文件不存在: {backend_file}")
        return
    
    if not frontend_file.exists():
        print(f"❌ 前端文件不存在: {frontend_file}")
        return
    
    print("✅ 文件检查通过")
    
    # 查找可用端口
    print("🔍 检查端口可用性...")
    backend_port = find_available_port(8000)
    frontend_port = find_available_port(8080)
    
    if not backend_port:
        print("❌ 无法找到可用的后端端口（尝试了8000-8099）")
        return
    
    if not frontend_port:
        print("❌ 无法找到可用的前端端口（尝试了8080-8179）")
        return
    
    if backend_port != 8000:
        print(f"⚠️ 端口8000被占用，使用端口{backend_port}")
    if frontend_port != 8080:
        print(f"⚠️ 端口8080被占用，使用端口{frontend_port}")
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    print(f"🚀 启动后端服务（端口{backend_port}）...")
    try:
        # 启动后端 - 不隐藏输出，便于调试
        backend_cmd = [
            sys.executable, "-m", "uvicorn", 
            "backend.api.main:app", 
            "--host", "0.0.0.0", 
            "--port", str(backend_port)
        ]
        
        print(f"执行命令: {' '.join(backend_cmd)}")
        backend_process = subprocess.Popen(backend_cmd)
        
        print(f"✅ 后端服务已启动 (PID: {backend_process.pid})")
        
        # 等待一会儿让后端启动
        time.sleep(3)
        
        # 检查后端是否成功启动
        if backend_process.poll() is not None:
            print("❌ 后端服务启动失败")
            return
        
        print(f"🌐 启动前端服务（端口{frontend_port}）...")
        
        # 切换到前端目录
        os.chdir(project_root / "frontend")
        
        # 启动前端
        frontend_cmd = [sys.executable, "-m", "http.server", str(frontend_port)]
        print(f"执行命令: {' '.join(frontend_cmd)}")
        frontend_process = subprocess.Popen(frontend_cmd)
        
        print(f"✅ 前端服务已启动 (PID: {frontend_process.pid})")
        
        # 等待前端启动
        time.sleep(2)
        
        # 检查前端是否成功启动
        if frontend_process.poll() is not None:
            print("❌ 前端服务启动失败")
            backend_process.terminate()
            return
        
        # 显示访问信息
        print("\n" + "=" * 50)
        print("🎉 NetView 已启动!")
        print("=" * 50)
        print(f"📊 后端API: http://localhost:{backend_port}")
        print(f"🌐 前端界面: http://localhost:{frontend_port}")
        print(f"📚 API文档: http://localhost:{backend_port}/docs")
        print("=" * 50)
        print("💡 注意事项:")
        print("  - 如果使用非默认端口，需要修改前端配置")
        print("  - 前端默认连接8000端口的后端API")
        print("  - 按 Ctrl+C 停止服务")
        print("=" * 50)
        
        # 如果后端端口不是8000，提醒用户修改前端配置
        if backend_port != 8000:
            print("⚠️ 重要提醒:")
            print(f"   后端运行在端口{backend_port}，但前端配置连接端口8000")
            print("   前端可能无法正常连接后端API")
            print("   建议：")
            print("   1. 停止占用8000端口的程序")
            print("   2. 或修改前端配置文件中的API地址")
        
        # 等待用户中断
        try:
            while True:
                time.sleep(1)
                # 检查进程是否还在运行
                if backend_process.poll() is not None:
                    print("⚠️ 后端进程已退出")
                    break
                if frontend_process.poll() is not None:
                    print("⚠️ 前端进程已退出")
                    break
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号...")
        
        # 停止服务
        print("正在停止服务...")
        try:
            backend_process.terminate()
            frontend_process.terminate()
            
            # 等待进程结束
            backend_process.wait(timeout=5)
            frontend_process.wait(timeout=5)
            
            print("✅ 服务已停止")
        except subprocess.TimeoutExpired:
            print("🔧 强制停止服务...")
            backend_process.kill()
            frontend_process.kill()
        except Exception as e:
            print(f"停止服务时出错: {e}")
        
        print("👋 NetView 已关闭")
        
    except FileNotFoundError:
        print("❌ 找不到 uvicorn 模块，请先安装依赖:")
        print("pip install fastapi uvicorn")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("请检查依赖是否已安装:")
        print("pip install -r requirements-minimal.txt")

if __name__ == "__main__":
    main()
