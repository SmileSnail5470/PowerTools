import logging
import logging.handlers
import os
from typing import Optional
from PySide6.QtCore import QObject, Signal, QMutex, QMutexLocker


class RotatingFileHandler(logging.handlers.RotatingFileHandler):    
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: Optional[str] = 'utf-8',
        delay: bool = False,
        errors: Optional[str] = None,
    ):
        """
        Args:
            filename: 日志文件路径
            mode: 文件打开模式
            max_bytes: 单个文件最大字节数
            backup_count: 保留的备份文件数量
            encoding: 文件编码
            delay: 是否延迟打开文件
            errors: 错误处理方式
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        super().__init__(
            filename=filename,
            mode=mode,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
            delay=delay,
            errors=errors,
        )
        
        self.max_bytes = max_bytes
        self.backup_count = backup_count
    
    def doRollover(self):
        super().doRollover()
        
        self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        base_path = self.baseFilename
        
        for i in range(self.backup_count + 1, self.backup_count + 10):
            old_file = f"{base_path}.{i}"
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                except OSError:
                    pass


class QtLogHandler(QObject, logging.Handler):
    
    log_message = Signal(str, str, str)  # message, level, formatted_message
    
    def __init__(self, parent: Optional[QObject] = None):
        """
        Args:
            parent: 父对象
        """
        QObject.__init__(self, parent)
        logging.Handler.__init__(self)
        
        self.mutex = QMutex()
        self._buffer = []
        self._max_buffer_size = 1000
    
    def emit(self, record: logging.LogRecord):
        try:
            with QMutexLocker(self.mutex):
                msg = self.format(record)
                level = record.levelname
                message = record.getMessage()
                
                self._buffer.append((message, level, msg))
                
                if len(self._buffer) > self._max_buffer_size:
                    self._buffer.pop(0)

                self.log_message.emit(message, level, msg)
        except Exception:
            self.handleError(record)
    
    def get_buffered_logs(self):
        with QMutexLocker(self.mutex):
            return self._buffer.copy()
    
    def clear_buffer(self):
        with QMutexLocker(self.mutex):
            self._buffer.clear()


class ErrorFileHandler(RotatingFileHandler): 
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 10,  # 错误日志保留更多备份
        encoding: Optional[str] = 'utf-8',
    ):
        super().__init__(filename, mode, max_bytes, backup_count, encoding)
        self.setLevel(logging.ERROR)


class PerformanceFileHandler(RotatingFileHandler):
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        max_bytes: int = 20 * 1024 * 1024,  # 性能日志可能较大
        backup_count: int = 3,
        encoding: Optional[str] = 'utf-8',
    ):
        super().__init__(filename, mode, max_bytes, backup_count, encoding)
        self.setLevel(logging.DEBUG)

