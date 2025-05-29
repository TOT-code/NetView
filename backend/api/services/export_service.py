"""
导出服务
处理各种格式的导出功能
"""

import json
import base64
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from .visualization_service import visualization_service

class ExportService:
    """导出服务类"""
    
    def __init__(self):
        self.visualization_service = visualization_service
        self.exports = {}  # 简单的内存存储，生产环境应使用数据库或文件系统
    
    def export_data(self, task_id: str, export_format: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """导出数据"""
        if options is None:
            options = {}
        
        # 获取可视化数据
        viz_data = self.visualization_service.get_visualization_data(task_id)
        if not viz_data:
            raise ValueError("任务不存在或未完成")
        
        # 生成导出ID
        export_id = f"export_{uuid.uuid4()}"
        
        # 根据格式处理导出
        if export_format == "json":
            result = self._export_json(viz_data, options)
        elif export_format == "png":
            result = self._export_png(viz_data, options)
        elif export_format == "svg":
            result = self._export_svg(viz_data, options)
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")
        
        # 保存导出记录
        export_record = {
            "export_id": export_id,
            "task_id": task_id,
            "export_format": export_format,
            "options": options,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24),
            "file_data": result["file_data"],
            "file_size": result["file_size"],
            "filename": result["filename"]
        }
        
        self.exports[export_id] = export_record
        
        return {
            "task_id": task_id,
            "export_format": export_format,
            "file_url": f"/api/v1/exports/download/{export_id}",
            "file_data": result["file_data"] if export_format == "json" else None,
            "file_size": result["file_size"],
            "filename": result["filename"],
            "expires_at": export_record["expires_at"]
        }
    
    def _export_json(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """导出JSON格式"""
        # 准备导出数据
        export_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "model_name": viz_data.get("model_name"),
                "graph_type": viz_data.get("graph_type"),
                "exporter": "NetView API v0.2.0"
            },
            "graph": {
                "nodes": viz_data.get("nodes", []),
                "edges": viz_data.get("edges", [])
            },
            "statistics": viz_data.get("statistics", {}),
            "enhanced_data": {}
        }
        
        # 添加增强数据
        if "dynamic_info" in viz_data:
            export_data["enhanced_data"]["dynamic_info"] = viz_data["dynamic_info"]
        
        if "architecture_patterns" in viz_data:
            export_data["enhanced_data"]["architecture_patterns"] = viz_data["architecture_patterns"]
        
        if "tensor_flow" in viz_data:
            export_data["enhanced_data"]["tensor_flow"] = viz_data["tensor_flow"]
        
        # 序列化为JSON
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        
        return {
            "file_data": json_str,
            "file_size": len(json_str.encode('utf-8')),
            "filename": f"{viz_data.get('model_name', 'model')}_analysis.json"
        }
    
    def _export_png(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """导出PNG格式"""
        # 模拟PNG生成（实际实现需要图形库如matplotlib或PIL）
        width = options.get("width", 1920)
        height = options.get("height", 1080)
        
        # 创建简单的PNG数据（实际应该渲染图形）
        png_data = self._create_mock_png(viz_data, width, height)
        
        # Base64编码
        png_base64 = base64.b64encode(png_data).decode('utf-8')
        
        return {
            "file_data": png_base64,
            "file_size": len(png_data),
            "filename": f"{viz_data.get('model_name', 'model')}_graph.png"
        }
    
    def _export_svg(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """导出SVG格式"""
        width = options.get("width", 1920)
        height = options.get("height", 1080)
        
        # 生成SVG内容
        svg_content = self._create_svg(viz_data, width, height)
        
        return {
            "file_data": svg_content,
            "file_size": len(svg_content.encode('utf-8')),
            "filename": f"{viz_data.get('model_name', 'model')}_graph.svg"
        }
    
    def _create_mock_png(self, viz_data: Dict[str, Any], width: int, height: int) -> bytes:
        """创建模拟PNG数据"""
        # 这里应该使用真实的图形库生成PNG
        # 现在返回一个简单的PNG头部作为示例
        png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x07\x80\x00\x00\x04\x38\x08\x02\x00\x00\x00'
        mock_data = b'\x00' * 1000  # 模拟图像数据
        return png_header + mock_data
    
    def _create_svg(self, viz_data: Dict[str, Any], width: int, height: int) -> str:
        """创建SVG内容"""
        nodes = viz_data.get("nodes", [])
        edges = viz_data.get("edges", [])
        
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <style>
        .node {{ fill: #e1f5fe; stroke: #0277bd; stroke-width: 2; }}
        .edge {{ stroke: #666; stroke-width: 1; marker-end: url(#arrowhead); }}
        .node-text {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
    </style>
    
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
        </marker>
    </defs>
    
    <!-- Model: {viz_data.get('model_name', 'Unknown')} -->
    <!-- Nodes: {len(nodes)}, Edges: {len(edges)} -->
    
'''
        
        # 添加边
        for i, edge in enumerate(edges):
            x1, y1 = self._get_node_position(edge.get("source"), i, len(edges), width, height)
            x2, y2 = self._get_node_position(edge.get("target"), i, len(edges), width, height)
            
            svg_content += f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="edge" />\n'
        
        # 添加节点
        for i, node in enumerate(nodes):
            x, y = self._get_node_position(node.get("id"), i, len(nodes), width, height)
            
            svg_content += f'''    <g>
        <circle cx="{x}" cy="{y}" r="30" class="node" />
        <text x="{x}" y="{y+5}" class="node-text">{node.get("type", "Node")}</text>
    </g>
'''
        
        svg_content += '</svg>'
        return svg_content
    
    def _get_node_position(self, node_id: str, index: int, total: int, width: int, height: int) -> tuple:
        """计算节点位置（简单布局算法）"""
        if total == 0:
            return width // 2, height // 2
        
        # 简单的网格布局
        cols = int((total ** 0.5)) + 1
        row = index // cols
        col = index % cols
        
        x = (col + 1) * width // (cols + 1)
        y = (row + 1) * height // (int(total / cols) + 2)
        
        return x, y
    
    def get_export_file(self, export_id: str) -> Optional[Dict[str, Any]]:
        """获取导出文件"""
        if export_id not in self.exports:
            return None
        
        export_record = self.exports[export_id]
        
        # 检查是否过期
        if datetime.now() > export_record["expires_at"]:
            del self.exports[export_id]
            return None
        
        return export_record
    
    def list_exports(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """列出导出记录"""
        exports_list = []
        
        for export_record in self.exports.values():
            if task_id and export_record["task_id"] != task_id:
                continue
            
            # 检查是否过期
            if datetime.now() > export_record["expires_at"]:
                continue
            
            exports_list.append({
                "export_id": export_record["export_id"],
                "task_id": export_record["task_id"],
                "export_format": export_record["export_format"],
                "filename": export_record["filename"],
                "file_size": export_record["file_size"],
                "created_at": export_record["created_at"],
                "expires_at": export_record["expires_at"]
            })
        
        return {
            "exports": exports_list,
            "total": len(exports_list)
        }
    
    def cleanup_expired_exports(self) -> int:
        """清理过期的导出文件"""
        current_time = datetime.now()
        expired_exports = []
        
        for export_id, export_record in self.exports.items():
            if current_time > export_record["expires_at"]:
                expired_exports.append(export_id)
        
        for export_id in expired_exports:
            del self.exports[export_id]
        
        return len(expired_exports)

# 创建单例服务实例
export_service = ExportService()
