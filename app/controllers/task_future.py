from enum import Enum
from typing import Optional, Any
from PySide6.QtCore import QObject, Signal, QMutex, QMutexLocker


class TaskStatus(Enum):
    PENDING = "pending"      # 已提交，尚未开始执行
    RUNNING = "running"      # 正在执行中
    DONE = "done"            # 成功完成
    FAILED = "failed"        # 执行过程中抛出异常
    CANCELLED = "cancelled"  # 被用户主动取消（若支持取消）


class TaskFuture(QObject):
    """任务Future对象，用于跟踪任务状态和结果"""
    
    progress = Signal(str, str)
    finished = Signal(object)
    failed = Signal(Exception)
    cancelled = Signal()
    state_changed = Signal(str)

    def __init__(self, job_id: str):
        super().__init__()
        self._job_id = job_id
        self._state = TaskStatus.PENDING.value
        self._result: Optional[Any] = None
        self._exception: Optional[Exception] = None
        self._cancel_requested = False
        self._mutex = QMutex()

    @property
    def job_id(self) -> str:
        """任务ID"""
        return self._job_id

    @property
    def state(self) -> str:
        """当前任务状态"""
        with QMutexLocker(self._mutex):
            return self._state

    @property
    def done(self) -> bool:
        """任务是否已完成（成功、失败或取消）"""
        with QMutexLocker(self._mutex):
            return self._state in (
                TaskStatus.DONE.value, 
                TaskStatus.FAILED.value, 
                TaskStatus.CANCELLED.value
            )

    def cancel(self) -> bool:
        """取消任务
        
        Returns:
            bool: 如果任务成功取消返回True，如果任务已完成则返回False
        """
        with QMutexLocker(self._mutex):
            if self._state in (
                TaskStatus.DONE.value,
                TaskStatus.FAILED.value,
                TaskStatus.CANCELLED.value
            ):
                return False
            self._cancel_requested = True
            self._state = TaskStatus.CANCELLED.value
        
        # 在锁外发送信号，避免死锁
        self.state_changed.emit(TaskStatus.CANCELLED.value)
        self.cancelled.emit()
        return True

    def cancelled_requested(self) -> bool:
        """检查是否请求了取消操作"""
        with QMutexLocker(self._mutex):
            return self._cancel_requested

    def set_state(self, state: str):
        """设置任务状态（内部使用）"""
        with QMutexLocker(self._mutex):
            if self._state in (
                TaskStatus.DONE.value,
                TaskStatus.FAILED.value,
                TaskStatus.CANCELLED.value
            ):
                return  # 已完成的任务不能改变状态
            self._state = state
        
        self.state_changed.emit(state)

    def set_progress(self, value: str, msg: str = ""):
        """更新任务进度
        
        Args:
            value: 进度值
            msg: 进度消息
        """
        self.progress.emit(value, msg)

    def set_result(self, result: Any):
        """设置任务结果（内部使用）"""
        with QMutexLocker(self._mutex):
            if self._state in (
                TaskStatus.DONE.value,
                TaskStatus.FAILED.value,
                TaskStatus.CANCELLED.value
            ):
                return  # 已完成的任务不能设置结果
            self._state = TaskStatus.DONE.value
            self._result = result
        
        self.state_changed.emit(TaskStatus.DONE.value)
        self.finished.emit(result)

    def set_exception(self, exc: Exception):
        """设置任务异常（内部使用）"""
        with QMutexLocker(self._mutex):
            if self._state in (
                TaskStatus.DONE.value,
                TaskStatus.FAILED.value,
                TaskStatus.CANCELLED.value
            ):
                return  # 已完成的任务不能设置异常
            self._state = TaskStatus.FAILED.value
            self._exception = exc
        
        self.state_changed.emit(TaskStatus.FAILED.value)
        self.failed.emit(exc)

    def result(self) -> Any:
        """获取任务结果
        
        Returns:
            任务执行结果
            
        Raises:
            RuntimeError: 如果任务尚未完成
            Exception: 如果任务执行失败，抛出原始异常
        """
        with QMutexLocker(self._mutex):
            if not self.done:
                raise RuntimeError("Task is not done yet.")
            if self._exception:
                raise self._exception
            return self._result