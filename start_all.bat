@echo off
chcp 65001 > nul
echo NetView 综合启动脚本
echo.

REM 设置项目根目录 (脚本的当前目录)
set PROJECT_DIR=%~dp0
echo 项目根目录: %PROJECT_DIR%
echo.

echo 检查并安装Python依赖...
echo 这可能需要一些时间，请稍候...
pip install -r requirements-minimal.txt

echo.
echo 启动后端 FastAPI 服务...
start "NetView Backend" /D "%PROJECT_DIR%" cmd /c "chcp 65001 > nul && python backend/api/main.py && pause || (echo 启动失败，按任意键查看错误... && pause)"

echo.
echo 启动前端 HTTP 服务器...
start "NetView Frontend" /D "%PROJECT_DIR%" cmd /c "chcp 65001 > nul && python -m http.server 8080"

echo.
echo 等待服务启动 (约 10 秒)...
timeout /t 10 /nobreak > nul

echo.
echo 在浏览器中打开前端界面...
start "" "http://localhost:8080/frontend/index.html"

echo.
echo 启动完成！
echo 后端 API 服务: http://localhost:8001
echo 前端界面: http://localhost:8080/frontend/index.html
echo.
echo 您可以关闭此脚本窗口，服务将继续在其他窗口中运行。
echo 要停止服务，请关闭对应的命令行窗口。

timeout /t 15 /nobreak > nul
