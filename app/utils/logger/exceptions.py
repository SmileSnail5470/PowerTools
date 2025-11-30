import logging
import sys
from typing import Optional, Callable, Any


class LoggedException(Exception):
    def __init__(self, message: str, logger: Optional[logging.Logger] = None, log_level: int = logging.ERROR):
        super().__init__(message)
        self.logger = logger or logging.getLogger(__name__)
        self.log_level = log_level
        self.logger.log(log_level, f"LoggedException: {message}", exc_info=True)


def log_unhandled_exceptions(logger: Optional[logging.Logger] = None):
    log = logger or logging.getLogger()
    
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        log.critical(
            "Not Catch Exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = exception_handler


def safe_call(
    func: Callable,
    default_return: Any = None,
    logger: Optional[logging.Logger] = None,
    *args,
    **kwargs
) -> Any:
    """安全调用函数，捕获所有异常
    
    Args:
        func: 要调用的函数
        default_return: 发生异常时的默认返回值
        logger: 日志记录器
        *args: 位置参数
        **kwargs: 关键字参数
    
    Returns:
        函数返回值或默认返回值
    """
    log = logger or logging.getLogger(func.__module__)
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log.error(
            f"Called function {func.__name__} exception: {e}",
            exc_info=True,
            extra={
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs,
            }
        )
        return default_return

