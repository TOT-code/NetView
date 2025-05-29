"""
模型分析服务
处理模型分析的核心业务逻辑
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.modules.parser import EnhancedModelExtractor
from backend.api.schemas.request import ModelAnalysisRequest, AnalysisOptions
from backend.api.schemas.response import TaskStatus

class ModelAnalysisService:
    """模型分析服务类"""
    
    def __init__(self):
        self.extractor = EnhancedModelExtractor()
        self.tasks = {}  # 简单的内存存储，生产环境应使用数据库
        
    async def create_analysis_task(self, request: ModelAnalysisRequest) -> Dict[str, Any]:
        """创建模型分析任务"""
        task_id = f"task_{uuid.uuid4()}"
        
        # 创建任务记录
        task = {
            "task_id": task_id,
            "status": TaskStatus.PENDING,
            "message": "任务已创建，等待处理",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "completed_at": None,
            "request": request.dict(),
            "result": None,
            "error": None,
            "progress": 0.0
        }
        
        self.tasks[task_id] = task
        
        # 异步启动分析任务
        asyncio.create_task(self._process_analysis(task_id, request))
        
        # 估算完成时间
        estimated_duration = self._estimate_duration(request)
        
        return {
            "task_id": task_id,
            "status": TaskStatus.PENDING,
            "message": "任务已创建，开始分析模型",
            "created_at": task["created_at"],
            "estimated_duration": estimated_duration
        }
    
    async def _process_analysis(self, task_id: str, request: ModelAnalysisRequest):
        """处理模型分析（异步）"""
        try:
            # 更新任务状态
            await self._update_task_status(task_id, TaskStatus.PROCESSING, "正在分析模型结构...", 10.0)
            
            # 执行模型分析
            await asyncio.sleep(0.1)  # 模拟异步处理
            
            # 调用增强型提取器
            enhanced_info = self.extractor.extract_from_code(
                request.code, 
                tuple(request.input_shape) if request.input_shape else None
            )
            
            await self._update_task_status(task_id, TaskStatus.PROCESSING, "生成可视化数据...", 80.0)
            
            # 生成可视化数据
            viz_data = self.extractor.get_enhanced_visualization_data(enhanced_info)
            
            await self._update_task_status(task_id, TaskStatus.PROCESSING, "完成分析...", 95.0)
            
            # 准备结果数据
            result = {
                "enhanced_info": self.extractor.to_dict(enhanced_info),
                "visualization_data": viz_data,
                "analysis_summary": self._create_analysis_summary(enhanced_info)
            }
            
            # 更新任务完成状态
            await self._update_task_status(
                task_id, 
                TaskStatus.COMPLETED, 
                "模型分析完成", 
                100.0,
                result=result
            )
            
        except Exception as e:
            # 处理错误
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "details": {
                    "code_length": len(request.code),
                    "input_shape": request.input_shape,
                    "analysis_options": request.analysis_options.dict() if request.analysis_options else None
                }
            }
            
            await self._update_task_status(
                task_id, 
                TaskStatus.FAILED, 
                f"分析失败: {str(e)}", 
                0.0,
                error=error_info
            )
    
    async def _update_task_status(self, task_id: str, status: TaskStatus, message: str, 
                                 progress: float, result: Optional[Dict] = None, 
                                 error: Optional[Dict] = None):
        """更新任务状态"""
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        task["status"] = status
        task["message"] = message
        task["progress"] = progress
        task["updated_at"] = datetime.now()
        
        if result:
            task["result"] = result
            
        if error:
            task["error"] = error
            
        if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED:
            task["completed_at"] = datetime.now()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if task_id not in self.tasks:
            return None
            
        task = self.tasks[task_id]
        
        result = {
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"],
            "created_at": task["created_at"],
            "updated_at": task["updated_at"],
            "completed_at": task["completed_at"]
        }
        
        if task["error"]:
            result["error"] = task["error"]["error"]
            result["error_details"] = task["error"]
            
        if task["status"] == TaskStatus.COMPLETED:
            result["result_url"] = f"/api/v1/models/{task_id}/visualization"
            
        return result
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        if task_id not in self.tasks:
            return None
            
        task = self.tasks[task_id]
        
        if task["status"] != TaskStatus.COMPLETED:
            return None
            
        return task["result"]
    
    def _estimate_duration(self, request: ModelAnalysisRequest) -> int:
        """估算分析时间"""
        base_time = 15  # 基础时间15秒
        
        # 根据代码长度调整
        code_factor = min(len(request.code) / 1000, 3)  # 最多增加3倍
        
        # 根据分析选项调整
        options_factor = 1.0
        if request.analysis_options:
            if request.analysis_options.enable_dynamic:
                options_factor += 0.3
            if request.analysis_options.enable_patterns:
                options_factor += 0.2
            if request.analysis_options.enable_tensor_flow:
                options_factor += 0.4
        
        return int(base_time * (1 + code_factor) * options_factor)
    
    def _create_analysis_summary(self, enhanced_info) -> Dict[str, Any]:
        """创建分析摘要"""
        summary = {
            "model_name": enhanced_info.model_name,
            "graph_type": enhanced_info.network_graph.graph_type,
            "node_count": len(enhanced_info.network_graph.nodes),
            "edge_count": len(enhanced_info.network_graph.edges),
            "complexity_score": self._calculate_complexity_score(enhanced_info),
            "features": []
        }
        
        # 添加特性标记
        if enhanced_info.dynamic_analysis and enhanced_info.dynamic_analysis.has_dynamic_control:
            summary["features"].append("动态控制流")
            
        if enhanced_info.architecture_patterns:
            summary["features"].append(f"架构模式: {enhanced_info.architecture_patterns.pattern_type}")
            
        if enhanced_info.tensor_flow_analysis:
            summary["features"].append(f"Tensor操作: {len(enhanced_info.tensor_flow_analysis.tensor_operations)}")
            
        return summary
    
    def _calculate_complexity_score(self, enhanced_info) -> float:
        """计算复杂度分数"""
        score = 0.0
        
        # 基于节点数
        score += len(enhanced_info.network_graph.nodes) * 0.1
        
        # 基于连接复杂度
        score += len(enhanced_info.network_graph.edges) * 0.05
        
        # 动态特性加分
        if enhanced_info.dynamic_analysis and enhanced_info.dynamic_analysis.has_dynamic_control:
            score += 2.0
            
        # 架构模式加分
        if enhanced_info.architecture_patterns:
            if enhanced_info.architecture_patterns.residual_connections:
                score += len(enhanced_info.architecture_patterns.residual_connections) * 0.5
                
        # 限制在0-10范围内
        return min(max(score, 0.0), 10.0)
    
    def list_tasks(self, limit: int = 50) -> Dict[str, Any]:
        """列出任务"""
        sorted_tasks = sorted(
            self.tasks.values(), 
            key=lambda x: x["created_at"], 
            reverse=True
        )[:limit]
        
        task_list = []
        for task in sorted_tasks:
            task_info = {
                "task_id": task["task_id"],
                "status": task["status"],
                "message": task["message"],
                "created_at": task["created_at"],
                "progress": task["progress"]
            }
            
            if task["result"]:
                task_info["model_name"] = task["result"]["enhanced_info"]["model_name"]
                
            task_list.append(task_info)
        
        return {
            "tasks": task_list,
            "total": len(self.tasks),
            "limit": limit
        }
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """清理旧任务"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if task["created_at"] < cutoff_time:
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            
        return len(tasks_to_remove)

# 创建单例服务实例
model_analysis_service = ModelAnalysisService()
