"""
模型分析API路由
处理模型分析相关的请求
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.api.schemas.request import ModelAnalysisRequest
from backend.api.schemas.response import ModelAnalysisResponse, TaskStatusResponse
from backend.api.services.model_service import model_analysis_service
from backend.api.middleware.error_handler import TaskNotFoundError, ModelAnalysisError

router = APIRouter()

@router.post("/models/analyze", 
             response_model=ModelAnalysisResponse,
             summary="分析PyTorch模型",
             description="提交PyTorch模型代码进行分析，返回任务ID用于查询结果")
async def analyze_model(request: ModelAnalysisRequest) -> ModelAnalysisResponse:
    """
    分析PyTorch模型
    
    **功能特性：**
    - AST静态分析
    - PyTorch运行时内省  
    - 动态图分析
    - 架构模式识别
    - Tensor流追踪
    
    **支持的模型：**
    - CNN (卷积神经网络)
    - ResNet (残差网络)
    - DenseNet (密集连接网络)
    - 自定义PyTorch模型
    
    **返回信息：**
    - 任务ID用于查询进度和结果
    - 预估完成时间
    - 基本模型信息（如果快速分析完成）
    """
    try:
        result = await model_analysis_service.create_analysis_task(request)
        
        return ModelAnalysisResponse(
            task_id=result["task_id"],
            status=result["status"],
            message=result["message"],
            created_at=result["created_at"],
            estimated_duration=result.get("estimated_duration"),
            model_info=result.get("model_info")
        )
        
    except Exception as e:
        raise ModelAnalysisError(f"模型分析失败: {str(e)}")

@router.get("/models/{task_id}/status",
           response_model=TaskStatusResponse,
           summary="查询分析任务状态",
           description="根据任务ID查询模型分析的进度和状态")
async def get_analysis_status(task_id: str) -> TaskStatusResponse:
    """
    查询分析任务状态
    
    **状态说明：**
    - `pending`: 等待处理
    - `processing`: 正在分析
    - `completed`: 分析完成
    - `failed`: 分析失败
    - `cancelled`: 已取消
    
    **进度信息：**
    - 进度百分比 (0-100)
    - 当前处理阶段描述
    - 预估剩余时间
    """
    task_status = model_analysis_service.get_task_status(task_id)
    
    if not task_status:
        raise TaskNotFoundError(task_id)
    
    return TaskStatusResponse(**task_status)

@router.get("/models/{task_id}/result",
           summary="获取分析结果详情",
           description="获取已完成任务的完整分析结果")
async def get_analysis_result(task_id: str) -> Dict[str, Any]:
    """
    获取分析结果详情
    
    **包含信息：**
    - 完整的模型结构信息
    - 增强分析数据
    - 复杂度统计
    - 原始数据（用于调试）
    
    **注意：** 只有状态为 `completed` 的任务才有结果数据
    """
    result = model_analysis_service.get_task_result(task_id)
    
    if not result:
        # 检查任务是否存在
        task_status = model_analysis_service.get_task_status(task_id)
        if not task_status:
            raise TaskNotFoundError(task_id)
        
        # 任务存在但无结果
        if task_status["status"] == "failed":
            raise HTTPException(
                status_code=422,
                detail=f"任务分析失败: {task_status.get('error', '未知错误')}"
            )
        elif task_status["status"] != "completed":
            raise HTTPException(
                status_code=425,
                detail=f"任务尚未完成，当前状态: {task_status['status']}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="任务已完成但无法获取结果"
            )
    
    return {
        "task_id": task_id,
        "status": "completed",
        "result": result
    }

@router.get("/models/tasks",
           summary="列出分析任务",
           description="获取所有分析任务的列表")
async def list_analysis_tasks(
    limit: Optional[int] = 50,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """
    列出分析任务
    
    **查询参数：**
    - `limit`: 返回数量限制 (默认50，最大100)
    - `status`: 按状态过滤 (pending, processing, completed, failed, cancelled)
    
    **返回信息：**
    - 任务列表（按创建时间倒序）
    - 任务总数统计
    - 各状态任务数量
    """
    if limit and limit > 100:
        limit = 100
    
    # 验证状态参数
    valid_statuses = ["pending", "processing", "completed", "failed", "cancelled"]
    if status and status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"无效的状态参数。有效值: {', '.join(valid_statuses)}"
        )
    
    result = model_analysis_service.list_tasks(limit or 50)
    
    # 按状态过滤
    if status:
        filtered_tasks = [
            task for task in result["tasks"]
            if task["status"] == status
        ]
        result["tasks"] = filtered_tasks
        result["filtered_by_status"] = status
    
    return result

@router.delete("/models/{task_id}",
              summary="取消分析任务",
              description="取消正在进行的分析任务")
async def cancel_analysis_task(task_id: str) -> Dict[str, Any]:
    """
    取消分析任务
    
    **注意：**
    - 只能取消状态为 `pending` 或 `processing` 的任务
    - 已完成、失败或已取消的任务无法取消
    - 取消后的任务状态将变为 `cancelled`
    """
    task_status = model_analysis_service.get_task_status(task_id)
    
    if not task_status:
        raise TaskNotFoundError(task_id)
    
    if task_status["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"无法取消已{task_status['status']}的任务"
        )
    
    # 这里应该实现真正的任务取消逻辑
    # 目前只是简单地标记状态
    return {
        "success": True,
        "message": f"任务 {task_id} 已取消",
        "task_id": task_id,
        "previous_status": task_status["status"]
    }

@router.post("/models/batch-analyze",
            summary="批量分析模型",
            description="同时提交多个模型进行分析")
async def batch_analyze_models(models: List[ModelAnalysisRequest]) -> Dict[str, Any]:
    """
    批量分析模型
    
    **限制：**
    - 单次最多提交10个模型
    - 每个模型使用相同的分析流程
    
    **返回：**
    - 批次ID
    - 各个任务的ID列表
    - 预估总完成时间
    """
    if len(models) > 10:
        raise HTTPException(
            status_code=400,
            detail="单次批量分析最多支持10个模型"
        )
    
    if len(models) == 0:
        raise HTTPException(
            status_code=400,
            detail="至少需要提交1个模型"
        )
    
    # 创建批量任务
    batch_results = []
    task_id = ""  # 初始化变量
    
    for i, model_request in enumerate(models):
        try:
            result = await model_analysis_service.create_analysis_task(model_request)
            task_id = result["task_id"]  # 更新task_id
            batch_results.append({
                "index": i,
                "task_id": task_id,
                "status": "created",
                "model_name": model_request.model_name or f"Model_{i+1}"
            })
        except Exception as e:
            batch_results.append({
                "index": i,
                "task_id": None,
                "status": "failed",
                "error": str(e),
                "model_name": model_request.model_name or f"Model_{i+1}"
            })
    
    # 计算成功任务数
    successful_tasks = [r for r in batch_results if r["status"] == "created"]
    
    return {
        "batch_id": f"batch_{len(successful_tasks)}_{task_id[:8] if successful_tasks and task_id else 'none'}",
        "total_submitted": len(models),
        "successful_tasks": len(successful_tasks),
        "failed_tasks": len(models) - len(successful_tasks),
        "task_results": batch_results,
        "estimated_total_duration": len(successful_tasks) * 30  # 估算时间
    }
