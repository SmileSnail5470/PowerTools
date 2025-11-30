import logging
import sys
from typing import Optional, Dict
from .config import LogConfig, LOG_LEVELS
from .handlers import (
    RotatingFileHandler,
    ErrorFileHandler,
    PerformanceFileHandler,
    QtLogHandler,
)
from .formatters import (
    DetailedFormatter,
    CompactFormatter,
    PerformanceFormatter,
    JSONFormatter,
)


class LogManager:
    def __init__(
        self,
        log_dir: Optional[str] = None,
        level: str = 'INFO',
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        """
        Args:
            log_dir: 日志文件目录
            level: 日志级别
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份文件数量
        """
        self.config = LogConfig(
            log_dir=log_dir,
            level=level,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
        
        self._handlers: Dict[str, logging.Handler] = {}
        self._qt_handler: Optional[QtLogHandler] = None
        
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(self.config.level_int)
        
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        if self.config.enable_file_logging:
            self._setup_file_handlers()
        
        if self.config.enable_console_logging:
            self._setup_console_handler()
        
        if self.config.enable_gui_logging:
            self._setup_gui_handler()
    
    def _setup_file_handlers(self):
        main_handler = RotatingFileHandler(
            filename=self.config.main_log_file,
            max_bytes=self.config.max_bytes,
            backup_count=self.config.backup_count,
            encoding='utf-8',
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(DetailedFormatter(use_colors=False))
        self._handlers['main_file'] = main_handler
        self.root_logger.addHandler(main_handler)
        
        error_handler = ErrorFileHandler(
            filename=self.config.error_log_file,
            max_bytes=self.config.max_bytes,
            backup_count=self.config.backup_count * 2,
            encoding='utf-8',
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(DetailedFormatter(use_colors=False))
        self._handlers['error_file'] = error_handler
        self.root_logger.addHandler(error_handler)
        
        perf_handler = PerformanceFileHandler(
            filename=self.config.performance_log_file,
            max_bytes=self.config.max_bytes * 2,
            backup_count=self.config.backup_count,
            encoding='utf-8',
        )
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.setFormatter(PerformanceFormatter())
        self._handlers['performance_file'] = perf_handler
        
        perf_logger = logging.getLogger('performance')
        perf_logger.setLevel(logging.DEBUG)
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False
    
    def _setup_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config.level_int)
        console_handler.setFormatter(DetailedFormatter(use_colors=True))
        self._handlers['console'] = console_handler
        self.root_logger.addHandler(console_handler)
    
    def _setup_gui_handler(self):
        self._qt_handler = QtLogHandler()
        self._qt_handler.setLevel(self.config.level_int)
        self._qt_handler.setFormatter(CompactFormatter())
        self._handlers['gui'] = self._qt_handler
        self.root_logger.addHandler(self._qt_handler)
    
    def get_qt_handler(self) -> Optional[QtLogHandler]:
        return self._qt_handler
    
    def update_level(self, level: str):
        """更新日志级别
        
        Args:
            level: 新的日志级别（字符串或整数）
        """
        if isinstance(level, str):
            level_int = LOG_LEVELS.get(level.upper(), logging.INFO)
        else:
            level_int = level
        
        self.config.update_level(level)
        self.root_logger.setLevel(level_int)
        
        for name, handler in self._handlers.items():
            if name not in ['main_file', 'error_file', 'performance_file']:
                handler.setLevel(level_int)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """获取日志记录器
        
        Args:
            name: 日志记录器名称，默认为调用模块名
        
        Returns:
            logging.Logger: 日志记录器实例
        """
        if name is None:
            import inspect
            frame = inspect.currentframe().f_back
            name = frame.f_globals.get('__name__', 'root')
        
        return logging.getLogger(name)
    
    def enable_console_logging(self, enable: bool = True):
        """启用或禁用控制台日志"""
        if enable and 'console' not in self._handlers:
            self._setup_console_handler()
        elif not enable and 'console' in self._handlers:
            handler = self._handlers.pop('console')
            self.root_logger.removeHandler(handler)
    
    def enable_file_logging(self, enable: bool = True):
        """启用或禁用文件日志"""
        if enable and 'main_file' not in self._handlers:
            self._setup_file_handlers()
        elif not enable:
            for name in ['main_file', 'error_file', 'performance_file']:
                if name in self._handlers:
                    handler = self._handlers.pop(name)
                    self.root_logger.removeHandler(handler)
    
    def enable_gui_logging(self, enable: bool = True):
        """启用或禁用GUI日志"""
        if enable and 'gui' not in self._handlers:
            self._setup_gui_handler()
        elif not enable and 'gui' in self._handlers:
            handler = self._handlers.pop('gui')
            self.root_logger.removeHandler(handler)
            if handler == self._qt_handler:
                self._qt_handler = None
    
    def cleanup_old_logs(self, days: int = 30):
        """清理指定天数之前的日志文件
        
        Args:
            days: 保留天数，默认30天
        """
        import os
        import time
        from pathlib import Path
        
        log_dir = Path(self.config.log_dir)
        if not log_dir.exists():
            return
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        deleted_count = 0
        
        for log_file in log_dir.glob('*.log*'):
            try:
                if os.path.getmtime(log_file) < cutoff_time:
                    os.remove(log_file)
                    deleted_count += 1
            except OSError:
                pass
        
        return deleted_count


def get_logger(name: str = None) -> logging.Logger:
    """获取日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
    
    Returns:
        logging.Logger: 日志记录器实例
    """
    import inspect
    
    if name is None:
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'root')
    
    return logging.getLogger(name)

