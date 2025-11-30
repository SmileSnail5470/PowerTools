import functools
import time
import traceback
import logging
from typing import Callable, Optional


def log_exception(
    logger: Optional[logging.Logger] = None,
    reraise: bool = False,
    log_args: bool = False,
    log_result: bool = False,
):
    """异常捕获和记录装饰器
    
    Args:
        logger: 日志记录器，如果为None则自动获取
        reraise: 是否重新抛出异常
        log_args: 是否记录函数参数
        log_result: 是否记录函数返回值
    
    Example:
        @log_exception(logger=logger, reraise=True)
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            
            func_name = f"{func.__module__}.{func.__name__}"
            
            if log_args:
                args_str = ', '.join([str(arg) for arg in args])
                kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
                log.debug(f"Called function {func_name}({args_str}, {kwargs_str})")
            else:
                log.debug(f"Called function {func_name}")
            
            try:
                result = func(*args, **kwargs)
                
                if log_result and result is not None:
                    result_str = str(result)
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    log.debug(f"Function {func_name} return: {result_str}")        
                return result
            except Exception as e:
                exc_type = type(e).__name__
                exc_msg = str(e)
                exc_tb = traceback.format_exc()
                
                log.error(
                    f"Function {func_name} exception: {exc_type}: {exc_msg}",
                    exc_info=True,
                    extra={
                        'function_name': func_name,
                        'exception_type': exc_type,
                        'exception_message': exc_msg,
                        'traceback': exc_tb,
                        'args': args if log_args else None,
                        'kwargs': kwargs if log_args else None,
                    }
                )              
                if reraise:
                    raise    
        return wrapper
    return decorator


def log_performance(
    logger: Optional[logging.Logger] = None,
    threshold: float = 0.1,
    log_args: bool = False,
    log_result: bool = False,
):
    """性能监控装饰器
    
    Args:
        logger: 日志记录器，如果为None则自动获取
        threshold: 时间阈值（秒），只记录超过此时间的函数调用
        log_args: 是否记录函数参数
        log_result: 是否记录函数返回值
    
    Example:
        @log_performance(logger=logger, threshold=0.5)
        def slow_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(f"{func.__module__}.performance")
            
            start_time = time.perf_counter()
            func_name = f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.perf_counter() - start_time
                
                if duration >= threshold:
                    extra = {
                        'duration': duration,
                        'function_name': func_name,
                        'threshold': threshold,
                    }
                    if log_args:
                        args_info = []
                        for i, arg in enumerate(args):
                            arg_str = str(arg)
                            if len(arg_str) > 100:
                                arg_str = arg_str[:100] + "..."
                            args_info.append(f"arg{i}={arg_str}")
                        for k, v in kwargs.items():
                            val_str = str(v)
                            if len(val_str) > 100:
                                val_str = val_str[:100] + "..."
                            args_info.append(f"{k}={val_str}")
                        extra['args_info'] = ', '.join(args_info)
                    if log_result and result is not None:
                        result_str = str(result)
                        if len(result_str) > 200:
                            result_str = result_str[:200] + "..."
                        extra['result_info'] = result_str
                    log.warning(
                        f"Function {func_name} cost {duration:.4f}s (threshold: {threshold}s)",
                        extra=extra
                    )
                
                return result            
            except Exception as e:
                duration = time.perf_counter() - start_time
                log.warning(
                    f"Function {func_name} exception, cost {duration:.4f}s",
                    extra={
                        'duration': duration,
                        'function_name': func_name,
                        'exception': str(e),
                    }
                )
                raise
        return wrapper
    return decorator


def log_function_call(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    log_args: bool = True,
    log_result: bool = False,
):
    """函数调用记录装饰器
    
    Args:
        logger: 日志记录器
        level: 日志级别
        log_args: 是否记录参数
        log_result: 是否记录返回值
    
    Example:
        @log_function_call(logger=logger, level=logging.INFO)
        def my_function(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            func_name = f"{func.__module__}.{func.__name__}"
            
            call_info = f"Called function {func_name}"
            
            if log_args:
                args_strs = []
                for i, arg in enumerate(args):
                    arg_str = str(arg)
                    if len(arg_str) > 100:
                        arg_str = arg_str[:100] + "..."
                    args_strs.append(f"arg{i}={arg_str}")
                
                for k, v in kwargs.items():
                    val_str = str(v)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    args_strs.append(f"{k}={val_str}")
                
                if args_strs:
                    call_info += f"({', '.join(args_strs)})"
            
            log.log(level, call_info)
            
            try:
                result = func(*args, **kwargs)
                
                if log_result and result is not None:
                    result_str = str(result)
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    log.log(level, f"Function {func_name} return: {result_str}")
                
                return result
            
            except Exception as e:
                log.error(f"Function {func_name} failed: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


class PerformanceTimer:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        message: str = "代码块执行",
        threshold: float = 0.0,
    ):
        """
        Args:
            logger: 日志记录器
            message: 描述信息
            threshold: 时间阈值，只记录超过此时间的代码块
        """
        self.logger = logger or logging.getLogger('performance')
        self.message = message
        self.threshold = threshold
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        
        if duration >= self.threshold:
            log_msg = f"{self.message} cost {duration:.4f}s"
            
            if exc_type is not None:
                self.logger.warning(
                    f"{log_msg} (Exception: {exc_type.__name__})",
                    extra={'duration': duration, 'exception': exc_type.__name__}
                )
            else:
                self.logger.info(log_msg, extra={'duration': duration})
        
        return False

