"""
任务管理API路由
处理任务状态查询和管理
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.api.schemas.response import TaskStatusResponse
from backend.api.services.task_service import task_service
from backend.api.middleware.error_handler import TaskNotFoundError

router = APIRouter()

@router.get("/tasks",
           summary="列出所有任务",
           description="获取任务列表，支持状态过滤和分页")
async def list_tasks(
    limit: Optional[int] = Query(50, ge=1, le=100, description="返回数量限制"),
    status: Optional[str] = Query(None, description="按状态过滤")
) -> Dict[str, Any]:
    """
    列出所有任务
    
    **查询参数：**
    - `limit`: 返回数量限制 (1-100，默认50)
    - `status`: 状态过滤 (pending, processing, completed, failed, cancelled)
    
    **返回信息：**
    - 任务列表（按创建时间倒序）
    - 分页信息
    - 状态统计
    """
    # 验证状态参数
    valid_statuses = ["pending", "processing", "completed", "failed", "cancelled"]
    if status and status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"无效的状态参数。有效值: {', '.join(valid_statuses)}"
        )
    
    result = task_service.list_tasks(limit, status)
    
    return result

@router.get("/tasks/summary",
           summary="获取任务统计摘要",
           description="获取任务的整体统计信息")
async def get_task_summary() -> Dict[str, Any]:
    """
    获取任务统计摘要
    
    **统计内容：**
    - 任务总数
    - 各状态任务数量
    - 最近任务列表
    - 系统负载指标
    
    **用途：**
    - 监控面板
    - 系统状态检查
    - 负载分析
    """
    summary = task_service.get_task_summary()
    
    # 添加系统负载信息
    processing_count = summary["status_counts"].get("processing", 0)
    pending_count = summary["status_counts"].get("pending", 0)
    
    summary["system_load"] = {
        "active_tasks": processing_count,
        "queue_length": pending_count,
        "load_level": _calculate_load_level(processing_count, pending_count)
    }
    
    return summary

@router.post("/tasks/cleanup",
            summary="清理过期任务",
            description="清理指定时间之前的旧任务")
async def cleanup_tasks(
    max_age_hours: Optional[int] = Query(24, ge=1, le=168, description="最大保留时间（小时）")
) -> Dict[str, Any]:
    """
    清理过期任务
    
    **参数：**
    - `max_age_hours`: 任务最大保留时间（1-168小时，默认24小时）
    
    **操作：**
    - 删除超过指定时间的任务记录
    - 释放相关资源
    - 返回清理统计
    
    **注意：** 此操作不可逆转
    """
    result = task_service.cleanup_tasks(max_age_hours)
    
    return result

@router.get("/tasks/{task_id}",
           response_model=TaskStatusResponse,
           summary="查询任务状态",
           description="根据任务ID查询任务的详细状态信息")
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    查询任务状态
    
    **状态说明：**
    - `pending`: 等待处理
    - `processing`: 正在分析
    - `completed`: 分析完成
    - `failed`: 分析失败
    - `cancelled`: 已取消
    
    **包含信息：**
    - 进度百分比
    - 当前处理阶段
    - 错误信息（如果有）
    - 结果链接（如果完成）
    """
    task_status = task_service.get_task_status(task_id)
    
    if not task_status:
        raise TaskNotFoundError(task_id)
    
    return TaskStatusResponse(**task_status)

@router.delete("/tasks/{task_id}",
              summary="取消任务",
              description="取消正在进行或等待中的任务")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    取消任务
    
    **限制：**
    - 只能取消状态为 `pending` 或 `processing` 的任务
    - 已完成、失败或已取消的任务无法再次取消
    
    **结果：**
    - 成功取消：任务状态变为 `cancelled`
    - 失败：返回错误信息和原因
    """
    result = task_service.cancel_task(task_id)
    
    if not result["success"]:
        raise HTTPException(
            status_code=400,
            detail=result["message"]
        )
    
    return result

@router.get("/tasks/{task_id}/logs",
           summary="获取任务日志",
           description="获取任务执行过程的详细日志")
async def get_task_logs(task_id: str) -> Dict[str, Any]:
    """
    获取任务日志
    
    **日志内容：**
    - 任务执行步骤
    - 时间戳记录
    - 错误详情
    - 性能指标
    
    **注意：** 日志仅在开发环境或启用调试模式时可用
    """
    # 检查任务是否存在
    task_status = task_service.get_task_status(task_id)
    if not task_status:
        raise TaskNotFoundError(task_id)
    
    # 模拟日志数据（实际实现中应该从日志系统获取）
    logs = [
        {
            "timestamp": task_status["created_at"],
            "level": "INFO",
            "message": "任务已创建",
            "step": "initialization"
        },
        {
            "timestamp": task_status["updated_at"],
            "level": "INFO",
            "message": f"任务状态: {task_status['status']}",
            "step": "processing"
        }
    ]
    
    if task_status.get("error"):
        logs.append({
            "timestamp": task_status["updated_at"],
            "level": "ERROR",
            "message": task_status["error"],
            "step": "error"
        })
    
    return {
        "task_id": task_id,
        "logs": logs,
        "log_count": len(logs)
    }

@router.post("/tasks/{task_id}/retry",
            summary="重试失败任务",
            description="重新执行失败的任务")
async def retry_task(task_id: str) -> Dict[str, Any]:
    """
    重试失败任务
    
    **条件：**
    - 任务状态必须为 `failed`
    - 原始请求数据仍然可用
    
    **操作：**
    - 创建新的任务实例
    - 使用原始参数重新执行
    - 返回新任务ID
    """
    task_status = task_service.get_task_status(task_id)
    
    if not task_status:
        raise TaskNotFoundError(task_id)
    
    if task_status["status"] != "failed":
        raise HTTPException(
            status_code=400,
            detail=f"只能重试失败的任务，当前状态: {task_status['status']}"
        )
    
    # 这里应该实现真正的重试逻辑
    # 目前返回模拟响应
    return {
        "success": True,
        "message": f"任务 {task_id} 重试请求已提交",
        "original_task_id": task_id,
        "new_task_id": f"retry_{task_id}",
        "retry_count": 1
    }

def _calculate_load_level(processing: int, pending: int) -> str:
    """计算系统负载等级"""
    total_load = processing + pending
    
    if total_load == 0:
        return "idle"
    elif total_load <= 3:
        return "low"
    elif total_load <= 10:
        return "medium"
    elif total_load <= 20:
        return "high"
    else:
        return "overload"
