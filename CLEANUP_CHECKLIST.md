# 🧹 NetView 可删除文件清单

## 📋 概述
以下是NetView项目中可以安全删除的测试脚本和临时文件，删除这些文件不会影响系统的核心功能。

## 🗑️ 可删除的文件

### 测试和演示文件
```
frontend/test_enhanced_graph.html          # 增强图形功能测试页面
frontend/test_*.html                       # 其他测试页面（如果存在）
```

### 临时开发文件
```
backend/api/services/enhanced_visualization_service.py    # 原始增强服务（不完整版本）
```

### 文档和说明文件
```
ENHANCED_VISUALIZATION_README.md           # 重构文档（可选删除）
CLEANUP_CHECKLIST.md                      # 本清单文件（删除后可删除）
README_*.md                               # 临时说明文件（如果存在）
```

### 开发工具和脚本
```
start_simple.py                           # 简单启动脚本（如果不需要）
test_*.py                                 # 测试脚本（如果存在）
debug_*.py                                # 调试脚本（如果存在）
```

### 临时和备份文件
```
*.bak                                     # 备份文件
*.tmp                                     # 临时文件
*.log                                     # 日志文件
*_backup.*                                # 备份文件
*_old.*                                   # 旧版本文件
.DS_Store                                 # macOS系统文件
Thumbs.db                                 # Windows系统文件
```

## ✅ 核心保留文件

### 前端核心文件（必须保留）
```
frontend/index.html                       # 主应用页面
frontend/script.js                        # 主应用脚本
frontend/enhanced-graph-manager.js        # 增强图形管理器
frontend/style.css                        # 样式文件
```

### 后端核心文件（必须保留）
```
backend/api/services/enhanced_visualization_service_complete.py  # 完整增强服务
backend/api/services/model_service.py     # 模型分析服务
backend/api/routers/visualization.py      # 可视化路由
backend/api/routers/models.py            # 模型路由
backend/api/schemas/                      # 数据模式定义
backend/api/middleware/                   # 中间件
backend/main.py                          # 后端入口
```

### 配置和依赖文件（必须保留）
```
requirements.txt                         # Python依赖
package.json                            # Node.js依赖（如果存在）
.gitignore                              # Git忽略文件
```

## 🔧 删除命令

### 批量删除命令（请谨慎使用）

#### Linux/macOS:
```bash
# 删除测试文件
rm -f frontend/test_*.html

# 删除不完整的增强服务
rm -f backend/api/services/enhanced_visualization_service.py

# 删除文档文件（可选）
rm -f ENHANCED_VISUALIZATION_README.md
rm -f CLEANUP_CHECKLIST.md

# 删除临时文件
find . -name "*.bak" -delete
find . -name "*.tmp" -delete
find . -name "*.log" -delete
find . -name "*_backup.*" -delete
find . -name "*_old.*" -delete
find . -name ".DS_Store" -delete
```

#### Windows PowerShell:
```powershell
# 删除测试文件
Remove-Item frontend\test_*.html -Force

# 删除不完整的增强服务
Remove-Item backend\api\services\enhanced_visualization_service.py -Force

# 删除文档文件（可选）
Remove-Item ENHANCED_VISUALIZATION_README.md -Force
Remove-Item CLEANUP_CHECKLIST.md -Force

# 删除临时文件
Get-ChildItem -Recurse -Name "*.bak" | Remove-Item -Force
Get-ChildItem -Recurse -Name "*.tmp" | Remove-Item -Force
Get-ChildItem -Recurse -Name "*.log" | Remove-Item -Force
Get-ChildItem -Recurse -Name "*_backup.*" | Remove-Item -Force
Get-ChildItem -Recurse -Name "*_old.*" | Remove-Item -Force
Get-ChildItem -Recurse -Name "Thumbs.db" | Remove-Item -Force
```

## ⚠️ 注意事项

1. **备份重要文件**: 删除前请确保已备份重要文件
2. **测试功能**: 删除后请测试系统核心功能是否正常
3. **版本控制**: 如果使用Git，建议提交当前状态后再删除
4. **团队协作**: 如果是团队项目，请与团队成员确认后再删除

## 📝 删除后验证

删除文件后，请验证以下功能：

- [ ] 主应用可以正常启动 (`frontend/index.html`)
- [ ] 后端API服务可以正常运行
- [ ] 模型分析和可视化功能正常
- [ ] 增强图形功能正常工作

## 🎯 推荐删除顺序

1. **第一步**: 删除明确的测试文件
   ```bash
   rm -f frontend/test_enhanced_graph.html
   ```

2. **第二步**: 删除不完整的服务文件
   ```bash
   rm -f backend/api/services/enhanced_visualization_service.py
   ```

3. **第三步**: 删除文档文件（可选）
   ```bash
   rm -f ENHANCED_VISUALIZATION_README.md
   rm -f CLEANUP_CHECKLIST.md
   ```

4. **第四步**: 清理临时文件
   ```bash
   find . -name "*.bak" -delete
   find . -name "*.tmp" -delete
   ```

执行删除后，NetView将保持核心功能完整，同时减少不必要的文件占用空间。
