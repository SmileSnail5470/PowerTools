import uuid
import logging
import threading
from typing import Optional, Dict, Callable
from PySide6.QtCore import QThreadPool, QObject, Signal
from app.controllers.worker import Worker
from app.controllers.task_future import TaskFuture
from app.utils.logger.decorators import log_function_call, log_exception
from app.ui.common.config import cfg


class TaskManager(QObject):
    """任务管理器，负责管理异步任务的提交、取消和生命周期"""
    
    all_done = Signal()

    def __init__(self, max_workers: int = 8, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(max_workers)
        self._tasks: Dict[str, TaskFuture] = {}
        self._lock = threading.Lock()
        self._shutting_down = False

    @log_exception(logger=logging.getLogger(), log_args=False, log_result=False)
    @log_function_call(logger=logging.getLogger(), level=logging.INFO, log_args=True, log_result=False)
    def submit(self, func: Callable, *args, **kwargs) -> TaskFuture:
        """提交一个新任务
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            TaskFuture: 任务未来对象
            
        Raises:
            RuntimeError: 如果任务管理器正在关闭
        """
        if self._shutting_down:
            raise RuntimeError("TaskManager is shutting down, cannot submit new tasks.")

        job_id = uuid.uuid4().hex
        future = TaskFuture(job_id)
        worker = Worker(func, future, *args, **kwargs)

        with self._lock:
            self._tasks[job_id] = future

        self.pool.start(worker)
        return future

    def cancel(self, job_id: str) -> bool:
        """取消指定的任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            bool: 如果任务存在且成功取消返回True，否则返回False
        """
        with self._lock:
            future = self._tasks.get(job_id)
            if future is None:
                return False
            # 在锁内获取future，但cancel操作本身是线程安全的
            return future.cancel()

    def get_future(self, job_id: str) -> Optional[TaskFuture]:
        """获取指定任务ID的Future对象
        
        Args:
            job_id: 任务ID
            
        Returns:
            TaskFuture或None
        """
        with self._lock:
            return self._tasks.get(job_id)

    def cleanup_completed_tasks(self) -> int:
        """清理已完成的任务，释放内存
        
        Returns:
            int: 清理的任务数量
        """
        with self._lock:
            completed_ids = [
                job_id for job_id, future in self._tasks.items()
                if future.done
            ]
            for job_id in completed_ids:
                del self._tasks[job_id]
            return len(completed_ids)

    def get_active_task_count(self) -> int:
        """获取当前活跃任务数量
        
        Returns:
            int: 活跃任务数量
        """
        with self._lock:
            return sum(1 for future in self._tasks.values() if not future.done)

    def shutdown(self, wait: bool = True, timeout: Optional[int] = None):
        """关闭任务管理器
        
        Args:
            wait: 是否等待所有任务完成
            timeout: 等待超时时间（毫秒），None表示无限等待
        """
        self._shutting_down = True

        # 在锁内获取所有future的副本，避免在遍历时字典被修改
        with self._lock:
            futures = list(self._tasks.values())
        
        # 在锁外执行cancel操作，避免死锁
        for future in futures:
            future.cancel()

        if wait:
            self.pool.waitForDone(timeout if timeout is not None else -1)

        with self._lock:
            self._tasks.clear()

    @log_function_call(logger=logging.getLogger(), level=logging.INFO, log_args=False, log_result=False)
    def close(self):
        """关闭并清理任务管理器"""
        if not self._shutting_down:
            self.shutdown(wait=True)
        if self.pool is not None:
            self.pool.clear()


global_task_manager = TaskManager(max_workers=int(cfg.get(cfg.taskParallelNumber)))


class InternalTaskManager:
    _instance = None

    @classmethod
    def get_pool(cls) -> QThreadPool:
        if cls._instance is None:
            cls._instance = QThreadPool()
            cls._instance.setMaxThreadCount(3)
        return cls._instance

    @classmethod
    def shutdown(cls):
        if cls._instance is not None:
            cls._instance.waitForDone()
            cls._instance = None