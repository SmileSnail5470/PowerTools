from enum import Enum
from PySide6.QtCore import QObject, Signal, QMutex, QMutexLocker


class TaskStatus(Enum):
    PENDING = "pending"      # 已提交，尚未开始执行
    RUNNING = "running"      # 正在执行中
    DONE = "done"            # 成功完成
    FAILED = "failed"        # 执行过程中抛出异常
    CANCELLED = "cancelled"  # 被用户主动取消（若支持取消）


class TaskFuture(QObject):
    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(Exception)
    cancelled = Signal()
    state_changed = Signal(str)

    def __init__(self, job_id):
        super().__init__()
        self._job_id = job_id
        self._state = TaskStatus.PENDING.value
        self._result = None
        self._exception = None
        self._cancel_requested = False
        self._mutex = QMutex()

    @property
    def job_id(self):
        return self._job_id

    @property
    def state(self):
        return self._state

    @property
    def done(self):
        return self._state in (
            TaskStatus.DONE.value, 
            TaskStatus.FAILED.value, 
            TaskStatus.CANCELLED.value
            )

    def cancel(self):
        with QMutexLocker(self._mutex):
            if not self.done:
                self._cancel_requested = True
                self._state = TaskStatus.CANCELLED.value
                self.state_changed.emit(TaskStatus.CANCELLED.value)
                self.cancelled.emit()
                return True
        return False

    def cancelled_requested(self):
        with QMutexLocker(self._mutex):
            return self._cancel_requested

    def set_state(self, state):
        self._state = state
        self.state_changed.emit(state)

    def set_progress(self, value: int):
        self.progress.emit(value)

    def set_result(self, result):
        if not self.done:
            self._state = TaskStatus.DONE.value
            self._result = result
            self.state_changed.emit(TaskStatus.DONE.value)
            self.finished.emit(result)

    def set_exception(self, exc: Exception):
        if not self.done:
            self._state = TaskStatus.FAILED.value
            self._exception = exc
            self.state_changed.emit(TaskStatus.FAILED.value)
            self.failed.emit(exc)

    def result(self):
        if self._exception:
            raise self._exception
        return self._result