from PySide6.QtCore import QRunnable, Slot
from typing import Callable, Any, Optional
from app.controllers.task_future import TaskStatus, TaskFuture
import traceback


class Worker(QRunnable):
    """工作线程，执行实际的任务函数"""
    
    def __init__(self, func: Callable, future: TaskFuture, *args, **kwargs):
        super().__init__()
        self.func = func
        self.future = future
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        """执行任务函数"""
        # 检查任务是否已被取消
        if self.future.state == TaskStatus.CANCELLED.value:
            return
            
        try:
            self.future.set_state(TaskStatus.RUNNING.value)

            def progress_cb(v: int):
                """进度回调函数"""
                if self.future.cancelled_requested():
                    raise InterruptedError("Task was cancelled.")
                self.future.set_progress(v)

            # 执行任务函数
            result = self.func(
                *self.args,
                progress_cb=progress_cb,
                cancel_requested=self.future.cancelled_requested,
                **self.kwargs
            )
            
            # 再次检查是否被取消
            if self.future.cancelled_requested():
                raise InterruptedError("Task was cancelled.")

            self.future.set_result(result)
            
        except InterruptedError:
            # 任务被取消，确保状态正确
            if not self.future.done:
                self.future.cancel()
        except Exception as e:
            # 记录异常堆栈
            traceback.print_exc()
            self.future.set_exception(e)