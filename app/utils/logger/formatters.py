import logging
import traceback
from datetime import datetime


class DetailedFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 品红色
        'RESET': '\033[0m',       # 重置
    }
    def __init__(self, use_colors: bool = False, include_traceback: bool = True):
        """
        Args:
            use_colors: 是否使用颜色（仅对控制台输出有效）
            include_traceback: 是否包含堆栈跟踪
        """
        super().__init__()
        self.use_colors = use_colors
        self.include_traceback = include_traceback
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        level_name = record.levelname
        color_start = self.COLORS.get(level_name, '') if self.use_colors else ''
        color_end = self.COLORS.get('RESET', '') if self.use_colors else ''
        
        parts = [
            f"{color_start}[{timestamp}]{color_end}",
            f"{color_start}[{level_name}]{color_end}",
            f"[{record.name}]",
        ]
        
        if hasattr(record, 'threadName') and record.threadName:
            parts.append(f"[Thread:{record.threadName}]")
        
        if hasattr(record, 'processName') and record.processName:
            parts.append(f"[Process:{record.processName}]")
        
        if record.pathname:
            filename = record.pathname.split('/')[-1].split('\\')[-1]
            parts.append(f"[{filename}:{record.lineno}]")
        
        if record.funcName:
            parts.append(f"[{record.funcName}]")
        
        message = record.getMessage()
        parts.append(f": {message}")
        
        if record.exc_info and self.include_traceback:
            exc_text = self.formatException(record.exc_info)
            parts.append(f"\n{exc_text}")
        
        if hasattr(record, 'extra_info') and record.extra_info:
            parts.append(f"\nExtra info: {record.extra_info}")
        
        return ''.join(parts)


class CompactFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        level_name = record.levelname[:1]
        
        message = record.getMessage()
        
        result = f"[{timestamp}] [{level_name}] {message}"
        
        if record.exc_info:
            exc_type = record.exc_info[0].__name__ if record.exc_info[0] else 'Exception'
            result += f" - {exc_type}"
        
        return result


class PerformanceFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        duration = getattr(record, 'duration', None)
        function_name = getattr(record, 'function_name', record.funcName)
        args_info = getattr(record, 'args_info', '')
        result_info = getattr(record, 'result_info', '')
        
        parts = [f"[{timestamp}]", f"[PERF]", f"[{function_name}]"]
        
        if duration is not None:
            parts.append(f"Cost: {duration:.4f}s")
        
        if args_info:
            parts.append(f"Args: {args_info}")
        
        if result_info:
            parts.append(f"Result: {result_info}")
        
        if record.getMessage():
            parts.append(f"Message: {record.getMessage()}")
        
        return " | ".join(parts)


class JSONFormatter(logging.Formatter):
    def __init__(self):
        import json
        self.json = json
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'pathname': record.pathname,
        }
        if hasattr(record, 'threadName'):
            log_data['thread'] = record.threadName
        if hasattr(record, 'processName'):
            log_data['process'] = record.processName
        
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info),
            }
        
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        return self.json.dumps(log_data, ensure_ascii=False, default=str)

