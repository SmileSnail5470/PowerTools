import os
from typing import Optional
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL


LOG_LEVELS = {
    'DEBUG': DEBUG,
    'INFO': INFO,
    'WARNING': WARNING,
    'ERROR': ERROR,
    'CRITICAL': CRITICAL,
}

LOG_LEVEL_NAMES = {
    DEBUG: 'DEBUG',
    INFO: 'INFO',
    WARNING: 'WARNING',
    ERROR: 'ERROR',
    CRITICAL: 'CRITICAL',
}


class LogConfig:
    def __init__(
        self,
        log_dir: Optional[str] = None,
        level: str = 'INFO',
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_file_logging: bool = True,
        enable_console_logging: bool = False,
        enable_gui_logging: bool = True,
    ):
        if log_dir is None:
            raise Exception("Params log_dir is None.")
        self.log_dir = log_dir
        self.level = level.upper() if isinstance(level, str) else level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.enable_gui_logging = enable_gui_logging
        os.makedirs(self.log_dir, exist_ok=True)
    
    @property
    def level_int(self):
        """获取日志级别的整数值"""
        if isinstance(self.level, int):
            return self.level
        return LOG_LEVELS.get(self.level.upper(), INFO)
    
    @property
    def main_log_file(self):
        """主日志文件路径"""
        return os.path.join(self.log_dir, 'powertools.log')
    
    @property
    def error_log_file(self):
        """错误日志文件路径"""
        return os.path.join(self.log_dir, 'powertools_error.log')
    
    @property
    def performance_log_file(self):
        """性能日志文件路径"""
        return os.path.join(self.log_dir, 'powertools_performance.log')
    
    def update_level(self, level: str):
        """更新日志级别"""
        self.level = level.upper() if isinstance(level, str) else level

