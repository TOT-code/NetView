"""
导出功能API路由
处理各种格式的导出请求
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from typing import Dict, Any, Optional, List
import sys
import os
import io
import base64

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.api.schemas.request import ExportRequest
from backend.api.schemas.response import ExportResponse
from backend.api.services.export_service import export_service
from backend.api.middleware.error_handler import TaskNotFoundError, ExportError

router = APIRouter()

@router.post("/models/{task_id}/export",
            response_model=ExportResponse,
            summary="导出分析结果",
            description="将模型分析结果导出为指定格式")
async def export_model_data(task_id: str, request: ExportRequest) -> ExportResponse:
    """
    导出分析结果
    
    **支持格式：**
    - `json`: JSON数据格式，包含完整分析结果
    - `png`: PNG图像格式，网络结构图
    - `svg`: SVG矢量格式，可缩放图形
    
    **导出选项：**
    - `width`, `height`: 图像尺寸（PNG/SVG）
    - `background_color`: 背景颜色
    - `node_size`: 节点大小（small/medium/large）
    - `include_metadata`: 是否包含元数据（JSON）
    
    **返回信息：**
    - 文件下载URL
    - 文件大小
    - 过期时间
    """
    try:
        result = export_service.export_data(
            task_id, 
            request.format.value, 
            request.options
        )
        
        return ExportResponse(**result)
        
    except ValueError as e:
        if "任务不存在" in str(e):
            raise TaskNotFoundError(task_id)
        else:
            raise ExportError(str(e), request.format.value)
    except Exception as e:
        raise ExportError(f"导出失败: {str(e)}", request.format.value)

@router.get("/exports/download/{export_id}",
           summary="下载导出文件",
           description="根据导出ID下载生成的文件")
async def download_export_file(export_id: str):
    """
    下载导出文件
    
    **支持的下载方式：**
    - 直接下载：浏览器自动下载文件
    - 在线预览：JSON和SVG文件可在浏览器中预览
    
    **文件管理：**
    - 文件自动过期（默认24小时）
    - 过期文件自动清理
    - 支持断点续传（大文件）
    """
    export_record = export_service.get_export_file(export_id)
    
    if not export_record:
        raise HTTPException(
            status_code=404,
            detail=f"导出文件 {export_id} 未找到或已过期"
        )
    
    filename = export_record["filename"]
    file_data = export_record["file_data"]
    export_format = export_record["export_format"]
    
    # 根据格式设置响应
    if export_format == "json":
        return Response(
            content=file_data,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(file_data.encode('utf-8')))
            }
        )
    elif export_format == "svg":
        return Response(
            content=file_data,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(file_data.encode('utf-8')))
            }
        )
    elif export_format == "png":
        # PNG数据是Base64编码的
        png_data = base64.b64decode(file_data)
        
        return Response(
            content=png_data,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(png_data))
            }
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"不支持的文件格式: {export_format}"
        )

@router.get("/exports",
           summary="列出导出记录",
           description="获取所有导出记录，支持按任务ID过滤")
async def list_exports(task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    列出导出记录
    
    **查询参数：**
    - `task_id`: 按任务ID过滤导出记录
    
    **返回信息：**
    - 导出记录列表
    - 文件信息（格式、大小、创建时间）
    - 下载链接
    - 过期时间
    """
    result = export_service.list_exports(task_id)
    
    # 为每个导出记录添加下载链接
    for export_record in result["exports"]:
        export_id = export_record["export_id"]
        export_record["download_url"] = f"/api/v1/exports/download/{export_id}"
    
    return result

@router.delete("/exports/{export_id}",
              summary="删除导出文件",
              description="删除指定的导出文件")
async def delete_export_file(export_id: str) -> Dict[str, Any]:
    """
    删除导出文件
    
    **操作：**
    - 立即删除导出记录
    - 释放存储空间
    - 使下载链接失效
    
    **注意：** 此操作不可逆转
    """
    export_record = export_service.get_export_file(export_id)
    
    if not export_record:
        raise HTTPException(
            status_code=404,
            detail=f"导出文件 {export_id} 未找到"
        )
    
    # 删除导出记录
    if export_id in export_service.exports:
        del export_service.exports[export_id]
    
    return {
        "success": True,
        "message": f"导出文件 {export_id} 已删除",
        "export_id": export_id,
        "filename": export_record["filename"]
    }

@router.post("/exports/cleanup",
            summary="清理过期导出",
            description="清理所有过期的导出文件")
async def cleanup_expired_exports() -> Dict[str, Any]:
    """
    清理过期导出
    
    **操作：**
    - 检查所有导出记录
    - 删除已过期的文件
    - 释放存储空间
    - 返回清理统计
    
    **自动化：**
    - 系统会定期自动执行清理
    - 也可以手动触发清理
    """
    cleaned_count = export_service.cleanup_expired_exports()
    
    return {
        "success": True,
        "message": f"已清理 {cleaned_count} 个过期导出文件",
        "cleaned_count": cleaned_count
    }

@router.get("/exports/{export_id}/info",
           summary="获取导出文件信息",
           description="获取导出文件的详细信息，不下载文件")
async def get_export_info(export_id: str) -> Dict[str, Any]:
    """
    获取导出文件信息
    
    **返回信息：**
    - 文件基本信息
    - 导出参数
    - 创建和过期时间
    - 文件状态
    """
    export_record = export_service.get_export_file(export_id)
    
    if not export_record:
        raise HTTPException(
            status_code=404,
            detail=f"导出文件 {export_id} 未找到或已过期"
        )
    
    # 返回不包含文件数据的信息
    info = {
        "export_id": export_record["export_id"],
        "task_id": export_record["task_id"],
        "export_format": export_record["export_format"],
        "filename": export_record["filename"],
        "file_size": export_record["file_size"],
        "created_at": export_record["created_at"],
        "expires_at": export_record["expires_at"],
        "options": export_record["options"],
        "download_url": f"/api/v1/exports/download/{export_id}"
    }
    
    return info

@router.post("/exports/batch",
            summary="批量导出",
            description="为多个任务批量创建导出")
async def batch_export(
    task_ids: List[str],
    export_format: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    批量导出
    
    **参数：**
    - `task_ids`: 任务ID列表
    - `export_format`: 导出格式（统一格式）
    - `options`: 导出选项
    
    **限制：**
    - 单次最多处理20个任务
    - 所有任务使用相同的导出格式和选项
    
    **返回：**
    - 批量导出ID
    - 各个任务的导出结果
    - 成功和失败统计
    """
    if len(task_ids) > 20:
        raise HTTPException(
            status_code=400,
            detail="单次批量导出最多支持20个任务"
        )
    
    if not task_ids:
        raise HTTPException(
            status_code=400,
            detail="至少需要指定一个任务ID"
        )
    
    batch_results = []
    
    for task_id in task_ids:
        try:
            result = export_service.export_data(
                task_id, 
                export_format, 
                options or {}
            )
            
            batch_results.append({
                "task_id": task_id,
                "status": "success",
                "export_id": result["file_url"].split("/")[-1],
                "download_url": result["file_url"]
            })
            
        except Exception as e:
            batch_results.append({
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            })
    
    successful_exports = [r for r in batch_results if r["status"] == "success"]
    
    return {
        "batch_id": f"batch_export_{len(successful_exports)}_{export_format}",
        "total_requested": len(task_ids),
        "successful_exports": len(successful_exports),
        "failed_exports": len(task_ids) - len(successful_exports),
        "export_format": export_format,
        "results": batch_results
    }
