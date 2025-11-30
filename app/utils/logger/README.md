# 日志系统使用指南

## 概述

这是一个完整的日志系统，提供了以下功能：

- ✅ 多级别日志支持（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- ✅ 文件持久化存储和自动轮转
- ✅ GUI日志查看器
- ✅ 异常捕获和记录
- ✅ 性能监控
- ✅ 日志文件管理
- ✅ 集成到PySide6应用

## 快速开始

### 1. 初始化日志系统

在应用启动时（通常在 `main.py` 中）初始化日志系统：

```python
from app.utils.logger import init_logging, get_log_manager

# 初始化日志系统
log_manager = init_logging(
    log_dir=None,           # 日志目录，默认为 ~/.powertools/logs
    level='INFO',           # 日志级别
    max_bytes=10*1024*1024, # 单个文件最大10MB
    backup_count=5          # 保留5个备份文件
)

# 获取日志记录器
logger = log_manager.get_logger(__name__)
logger.info("应用启动")
```

### 2. 基本使用

```python
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 不同级别的日志
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

### 3. 使用装饰器

#### 异常捕获装饰器

```python
from app.utils.logger.decorators import log_exception

@log_exception(logger=logger, reraise=True)
def my_function(x, y):
    return x / y  # 如果y为0，会自动记录异常
```

#### 性能监控装饰器

```python
from app.utils.logger.decorators import log_performance

@log_performance(logger=logger, threshold=0.5)
def slow_function():
    import time
    time.sleep(1)  # 如果执行时间超过0.5秒，会记录性能日志
    return "done"
```

#### 函数调用记录装饰器

```python
from app.utils.logger.decorators import log_function_call

@log_function_call(logger=logger, level=logging.INFO)
def important_function(x, y):
    return x + y  # 每次调用都会记录
```

### 4. 使用性能计时器

```python
from app.utils.logger.decorators import PerformanceTimer

with PerformanceTimer(logger=logger, message="处理数据", threshold=0.1):
    # 你的代码
    process_data()
    # 如果执行时间超过0.1秒，会自动记录
```

### 5. GUI日志查看器

日志查看器已集成到主窗口中，可以通过导航栏访问。

如果需要在其他地方使用：

```python
from app.ui.widgets.log_viewer_widget import LogViewerWidget

log_viewer = LogViewerWidget(parent=your_widget)
layout.addWidget(log_viewer)
```

### 6. 动态更新日志级别

```python
from app.utils.logger import get_log_manager

log_manager = get_log_manager()
log_manager.update_level('DEBUG')  # 切换到DEBUG级别
```

### 7. 清理旧日志

```python
from app.utils.logger import get_log_manager

log_manager = get_log_manager()
deleted_count = log_manager.cleanup_old_logs(days=30)  # 删除30天前的日志
print(f"删除了 {deleted_count} 个旧日志文件")
```

## 日志文件位置

日志文件默认保存在 `~/.powertools/logs/` 目录下：

- `powertools.log` - 所有日志
- `powertools_error.log` - 仅错误日志
- `powertools_performance.log` - 性能日志

## 日志格式

### 文件日志格式

```
[2024-01-01 12:00:00.123] [INFO    ] [module_name] [Thread:MainThread] [file.py:42] [function_name]: 日志消息
```

### GUI日志格式

```
[12:00:00.123] [I] 日志消息
```

## 最佳实践

1. **为每个模块创建独立的日志记录器**：
   ```python
   logger = get_logger(__name__)
   ```

2. **使用适当的日志级别**：
   - DEBUG: 详细的调试信息
   - INFO: 一般信息，记录程序执行流程
   - WARNING: 警告信息，程序可以继续运行
   - ERROR: 错误信息，影响功能但程序可以继续
   - CRITICAL: 严重错误，可能导致程序崩溃

3. **在关键位置添加异常记录**：
   ```python
   try:
       risky_operation()
   except Exception as e:
       logger.error("操作失败", exc_info=True)
   ```

4. **对性能敏感的函数使用性能监控**：
   ```python
   @log_performance(threshold=1.0)
   def expensive_operation():
       pass
   ```

5. **定期清理旧日志**：
   ```python
   # 在应用启动或定期任务中
   log_manager.cleanup_old_logs(days=30)
   ```

## 配置

日志级别可以通过应用配置系统动态调整：

```python
from app.ui.common.config import cfg

# 设置日志级别
cfg.logLevel.value = "DEBUG"

# 日志级别会实时生效
```

## 故障排查

如果日志没有正常记录，检查：

1. 日志系统是否已初始化（`init_logging()`）
2. 日志级别设置是否合适
3. 日志目录是否有写入权限
4. 检查日志文件是否存在且可读

