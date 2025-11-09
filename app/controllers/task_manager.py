from PySide6.QtCore import QThreadPool
from app.controllers.worker import Worker
from app.controllers.task_future import TaskFuture
import uuid


class TaskManager:
    def __init__(self, max_workers: int = 4):
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(max_workers)
        self._tasks = {}

    def submit(self, func, *args, **kwargs) -> TaskFuture:
        job_id = uuid.uuid4().hex
        future = TaskFuture(job_id)
        worker = Worker(func, future, *args, **kwargs)
        self._tasks[job_id] = future
        self.pool.start(worker)
        return future

    def cancel(self, job_id: str):
        if job_id in self._tasks:
            return self._tasks[job_id].cancel()
        return False

    def get_future(self, job_id: str):
        return self._tasks.get(job_id)

    def shutdown(self, wait=True):
        if wait:
            self.pool.waitForDone()