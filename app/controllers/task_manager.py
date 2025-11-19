import uuid
import threading
from PySide6.QtCore import QThreadPool, QObject, Signal
from app.controllers.worker import Worker
from app.controllers.task_future import TaskFuture


class TaskManager(QObject):

    all_done = Signal()

    def __init__(self, max_workers: int = 8, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(max_workers)
        self._tasks = {}
        self._lock = threading.Lock()
        self._shutting_down = False

    def submit(self, func, *args, **kwargs) -> TaskFuture:
        if self._shutting_down:
            raise RuntimeError("TaskManager is shutting down, cannot submit new tasks.")

        job_id = uuid.uuid4().hex
        future = TaskFuture(job_id)
        worker = Worker(func, future, *args, **kwargs)

        with self._lock:
            self._tasks[job_id] = future

        self.pool.start(worker)
        return future

    def cancel(self, job_id: str):
        with self._lock:
            future = self._tasks.get(job_id)
        return future.cancel() if future else False

    def get_future(self, job_id: str):
        with self._lock:
            return self._tasks.get(job_id)

    def shutdown(self, wait=True, timeout=None):
        self._shutting_down = True

        for future in self._tasks.values():
            future.cancel()

        if wait:
            self.pool.waitForDone(timeout if timeout is not None else -1)

        with self._lock:
            self._tasks.clear()

    def close(self):
        self.shutdown(wait=True)
        self.pool.clear()
        self.pool = None


global_task_manager = TaskManager()