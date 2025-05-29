"""
错误处理中间件
统一处理API错误和异常
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from datetime import datetime
import traceback
import uuid
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_error_handlers(app: FastAPI):
    """添加错误处理器到FastAPI应用"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """处理HTTP异常"""
        request_id = str(uuid.uuid4())[:8]
        
        error_response = {
            "error": "HTTPException",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": str(request.url)
        }
        
        # 记录错误日志
        logger.error(f"HTTP Exception [{request_id}]: {exc.status_code} - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """处理请求验证错误"""
        request_id = str(uuid.uuid4())[:8]
        
        # 提取验证错误详情
        validation_details = []
        for error in exc.errors():
            validation_details.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        error_response = {
            "error": "ValidationError",
            "message": "请求数据验证失败",
            "details": {
                "validation_errors": validation_details,
                "error_count": len(validation_details)
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": str(request.url)
        }
        
        # 记录错误日志
        logger.warning(f"Validation Error [{request_id}]: {len(validation_details)} errors in {request.url}")
        
        return JSONResponse(
            status_code=422,
            content=error_response
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """处理值错误"""
        request_id = str(uuid.uuid4())[:8]
        
        error_response = {
            "error": "ValueError",
            "message": str(exc),
            "details": {
                "error_type": "business_logic_error",
                "suggestion": "请检查输入参数是否正确"
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": str(request.url)
        }
        
        # 记录错误日志
        logger.warning(f"Value Error [{request_id}]: {str(exc)}")
        
        return JSONResponse(
            status_code=400,
            content=error_response
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        """处理文件未找到错误"""
        request_id = str(uuid.uuid4())[:8]
        
        error_response = {
            "error": "FileNotFoundError",
            "message": "请求的资源未找到",
            "details": {
                "resource": str(exc),
                "suggestion": "请检查资源ID是否正确或资源是否已过期"
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": str(request.url)
        }
        
        logger.warning(f"File Not Found [{request_id}]: {str(exc)}")
        
        return JSONResponse(
            status_code=404,
            content=error_response
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """处理一般异常"""
        request_id = str(uuid.uuid4())[:8]
        
        # 获取异常详情
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        
        # 开发环境下包含完整错误堆栈
        details = {
            "error_type": exc_type,
            "suggestion": "请联系技术支持或稍后重试"
        }
        
        # 在开发环境中添加堆栈跟踪
        import os
        if os.getenv("ENVIRONMENT", "development") == "development":
            details["traceback"] = traceback.format_exc()
        
        error_response = {
            "error": "InternalServerError",
            "message": "服务器内部错误",
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": str(request.url)
        }
        
        # 记录详细错误日志
        logger.error(f"Internal Error [{request_id}]: {exc_type} - {exc_msg}")
        logger.error(f"Traceback [{request_id}]:\n{traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )
    
    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        """添加请求ID中间件"""
        request_id = str(uuid.uuid4())[:8]
        
        # 将请求ID添加到请求状态
        request.state.request_id = request_id
        
        # 记录请求开始
        start_time = datetime.now()
        logger.info(f"Request [{request_id}] START: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            
            # 计算处理时间
            process_time = (datetime.now() - start_time).total_seconds()
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # 记录请求完成
            logger.info(f"Request [{request_id}] END: {response.status_code} ({process_time:.3f}s)")
            
            return response
            
        except Exception as exc:
            # 计算处理时间
            process_time = (datetime.now() - start_time).total_seconds()
            
            # 记录请求异常
            logger.error(f"Request [{request_id}] ERROR: {type(exc).__name__} ({process_time:.3f}s)")
            
            # 重新抛出异常让错误处理器处理
            raise exc

class APIError(Exception):
    """自定义API错误基类"""
    
    def __init__(self, message: str, status_code: int = 400, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ModelAnalysisError(APIError):
    """模型分析错误"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=422, details=details)

class TaskNotFoundError(APIError):
    """任务未找到错误"""
    
    def __init__(self, task_id: str):
        message = f"任务 {task_id} 未找到"
        details = {"task_id": task_id}
        super().__init__(message, status_code=404, details=details)

class ExportError(APIError):
    """导出错误"""
    
    def __init__(self, message: str, export_format: str = None):
        details = {"export_format": export_format} if export_format else {}
        super().__init__(message, status_code=422, details=details)

# 添加自定义错误处理器
def add_custom_error_handlers(app: FastAPI):
    """添加自定义错误处理器"""
    
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        """处理自定义API错误"""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        
        error_response = {
            "error": type(exc).__name__,
            "message": exc.message,
            "details": exc.details,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": str(request.url)
        }
        
        logger.warning(f"API Error [{request_id}]: {type(exc).__name__} - {exc.message}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )

# 更新主函数以包含自定义错误处理器
def add_all_error_handlers(app: FastAPI):
    """添加所有错误处理器"""
    add_error_handlers(app)
    add_custom_error_handlers(app)
