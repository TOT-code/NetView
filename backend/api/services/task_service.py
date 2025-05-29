"""
任务管理服务
处理任务状态查询和管理
"""

from typing import Dict, Any, Optional, List
from .model_service import model_analysis_service

class TaskService:
    """任务管理服务类"""
    
    def __init__(self):
        self.model_service = model_analysis_service
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        return self.model_service.get_task_status(task_id)
    
    def list_tasks(self, limit: int = 50, status: Optional[str] = None) -> Dict[str, Any]:
        """列出任务"""
        result = self.model_service.list_tasks(limit)
        
        # 如果指定了状态过滤
        if status:
            filtered_tasks = [
                task for task in result["tasks"]
                if task["status"] == status
            ]
            result["tasks"] = filtered_tasks
            result["filtered_by_status"] = status
        
        return result
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """取消任务"""
        task_status = self.model_service.get_task_status(task_id)
        
        if not task_status:
            return {
                "success": False,
                "message": "任务不存在"
            }
        
        if task_status["status"] in ["completed", "failed", "cancelled"]:
            return {
                "success": False,
                "message": f"任务已{task_status['status']}，无法取消"
            }
        
        # 更新任务状态为取消
        # 注意：这里需要访问内部方法，实际实现中应该通过正式接口
        if task_id in self.model_service.tasks:
            task = self.model_service.tasks[task_id]
            task["status"] = "cancelled"
            task["message"] = "任务已被用户取消"
            task["updated_at"] = task_status["updated_at"]
        
        return {
            "success": True,
            "message": "任务已取消",
            "task_id": task_id
        }
    
    def cleanup_tasks(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """清理过期任务"""
        cleaned_count = self.model_service.cleanup_old_tasks(max_age_hours)
        
        return {
            "success": True,
            "message": f"已清理{cleaned_count}个过期任务",
            "cleaned_count": cleaned_count,
            "max_age_hours": max_age_hours
        }
    
    def get_task_summary(self) -> Dict[str, Any]:
        """获取任务统计摘要"""
        all_tasks = self.model_service.list_tasks(limit=1000)["tasks"]
        
        summary = {
            "total_tasks": len(all_tasks),
            "status_counts": {},
            "recent_tasks": all_tasks[:10]  # 最近10个任务
        }
        
        # 统计各状态任务数量
        for task in all_tasks:
            status = task["status"]
            summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        
        return summary

# 创建单例服务实例
task_service = TaskService()
