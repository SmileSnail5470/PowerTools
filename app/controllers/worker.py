from PySide6.QtCore import QRunnable, Slot
from app.controllers.task_future import TaskStatus, TaskFuture
import traceback


class Worker(QRunnable):
    def __init__(self, func, future: TaskFuture, *args, **kwargs):
        super().__init__()
        self.func = func
        self.future = future
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        if self.future.state == TaskStatus.CANCELLED.value:
            return
        try:
            self.future.set_state(TaskStatus.RUNNING.value)

            def progress_cb(v: int):
                if self.future.cancelled_requested():
                    raise InterruptedError("Task was cancelled.")
                self.future.set_progress(v)

            result = self.func(
                *self.args,
                progress_cb=progress_cb,
                cancel_requested=self.future.cancelled_requested,
                **self.kwargs
            )
            if self.future.cancelled_requested():
                raise InterruptedError("Task was cancelled.")

            self.future.set_result(result)
        except InterruptedError:
            self.future.cancel()
        except Exception as e:
            traceback.print_exc()
            self.future.set_exception(e)