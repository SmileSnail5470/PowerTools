import multiprocessing
import queue
from typing import Callable
from PySide6.QtCore import QRunnable, Slot
from app.controllers.task_future import TaskStatus, TaskFuture


_MSG_PROGRESS = "progress"
_MSG_RESULT = "result"
_MSG_ERROR = "error"
_POLL_INTERVAL = 0.05


def _subprocess_target(func, msg_queue, cancel_event, args, kwargs):
    try:
        def progress_cb(v: str, msg: str):
            if cancel_event.is_set():
                raise InterruptedError("Task was cancelled.")
            msg_queue.put((_MSG_PROGRESS, v, msg))

        def cancel_requested():
            return cancel_event.is_set()

        result = func(*args, progress_cb=progress_cb, cancel_requested=cancel_requested, **kwargs)

        if cancel_event.is_set():
            raise InterruptedError("Task was cancelled.")

        msg_queue.put((_MSG_RESULT, result))
    except Exception as e:
        msg_queue.put((_MSG_ERROR, type(e).__name__, str(e)))


class Worker(QRunnable):
    def __init__(self, func: Callable, future: TaskFuture, *args, **kwargs):
        super().__init__()
        self.func = func
        self.future = future
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        if self.future.state == TaskStatus.CANCELLED.value:
            return

        self.future.set_state(TaskStatus.RUNNING.value)

        ctx = multiprocessing.get_context("spawn")
        msg_queue = ctx.Queue()
        cancel_event = ctx.Event()

        process = ctx.Process(
            target=_subprocess_target,
            args=(self.func, msg_queue, cancel_event, self.args, self.kwargs),
            daemon=True,
        )
        process.start()

        try:
            self._monitor_loop(process, msg_queue, cancel_event)
        finally:
            self._cleanup(process, msg_queue)

    def _monitor_loop(self, process, msg_queue, cancel_event):
        while process.is_alive():
            if self.future.cancelled_requested():
                cancel_event.set()
            self._drain_queue(msg_queue)
            process.join(timeout=_POLL_INTERVAL)
        self._drain_queue(msg_queue)
        if not self.future.done:
            if self.future.cancelled_requested():
                self.future.cancel()
            elif process.exitcode != 0:
                self.future.set_exception(RuntimeError(f"Subprocess exited with code {process.exitcode}"))

    def _drain_queue(self, msg_queue):
        while True:
            try:
                msg = msg_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_message(msg)

    def _handle_message(self, msg):
        msg_type = msg[0]
        if msg_type == _MSG_PROGRESS:
            self.future.set_progress(msg[1], msg[2])
        elif msg_type == _MSG_RESULT:
            self.future.set_result(msg[1])
        elif msg_type == _MSG_ERROR:
            exc_name, exc_msg = msg[1], msg[2]
            if exc_name == "InterruptedError":
                self.future.cancel()
            else:
                self.future.set_exception(RuntimeError(f"{exc_name}: {exc_msg}"))

    def _cleanup(self, process, msg_queue):
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        process.close()
        try:
            while True:
                msg_queue.get_nowait()
        except queue.Empty:
            pass
        msg_queue.close()
        msg_queue.join_thread()
