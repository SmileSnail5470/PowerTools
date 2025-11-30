from .manager import LogManager, get_logger
from .decorators import (
    log_exception,
    log_performance,
    log_function_call
)
from .handlers import QtLogHandler

__all__ = [
    'LogManager',
    'get_logger',
    'log_exception',
    'log_performance',
    'log_function_call',
    'QtLogHandler',
]

_log_manager: LogManager = None


def init_logging(log_dir=None, level='INFO', max_bytes=10*1024*1024, backup_count=5):
    global _log_manager
    _log_manager = LogManager(log_dir, level, max_bytes, backup_count)
    return _log_manager


def get_log_manager():
    global _log_manager
    if _log_manager is None:
        raise RuntimeError("Need to call init_logging()")
    return _log_manager

