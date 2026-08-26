import os
import sys
import time
import shutil
import tempfile
import logging
import multiprocessing
import queue
import traceback
from typing import Callable
from PySide6.QtCore import QRunnable, Slot
from app.utils.logger import get_log_manager
from app.controllers.task_future import TaskStatus, TaskFuture

logger = logging.getLogger(__name__)

_MSG_PROGRESS = "progress"
_MSG_RESULT = "result"
_MSG_ERROR = "error"
_POLL_INTERVAL = 0.05
_CANCEL_GRACE_SEC = 2.0


def _subprocess_target(func, msg_queue, cancel_event, log_dir, task_tmp_dir, args, kwargs):
    if task_tmp_dir:
        try:
            os.makedirs(task_tmp_dir, exist_ok=True)
            tempfile.tempdir = task_tmp_dir
            os.environ["POWERTOOLS_TASK_TMPDIR"] = task_tmp_dir
        except Exception:
            pass

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "powertools.log")
        fh = logging.FileHandler(log_file_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"))
        root_logger.addHandler(fh)
        # 重定向 stdout/stderr 到日志文件，捕获 print() 输出
        log_stream = open(log_file_path, "a", encoding="utf-8")
        sys.stdout = log_stream
        sys.stderr = log_stream
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
        tb = traceback.format_exc()
        logging.getLogger("subprocess").error(f"Subprocess exception: {type(e).__name__}: {e}\n{tb}")
        msg_queue.put((_MSG_ERROR, type(e).__name__, str(e)))


class Worker(QRunnable):
    def __init__(self, func: Callable, future: TaskFuture, *args, **kwargs):
        super().__init__()
        self.func = func
        self.future = future
        self.args = args
        self.kwargs = kwargs

    def _get_log_dir(self):
        try:
            return get_log_manager().config.log_dir
        except Exception:
            return None

    @Slot()
    def run(self):
        if self.future.state == TaskStatus.CANCELLED.value:
            return

        self.future.set_state(TaskStatus.RUNNING.value)

        try:
            task_tmp_dir = tempfile.mkdtemp(prefix=f"pt_task_{self.future.job_id}_")
        except Exception:
            task_tmp_dir = None

        ctx = multiprocessing.get_context("spawn")
        msg_queue = ctx.Queue()
        cancel_event = ctx.Event()

        process = ctx.Process(
            target=_subprocess_target,
            args=(self.func, msg_queue, cancel_event, self._get_log_dir(), task_tmp_dir, self.args, self.kwargs),
            daemon=True,
        )
        process.start()

        try:
            self._monitor_loop(process, msg_queue, cancel_event)
        finally:
            self._cleanup(process, msg_queue, task_tmp_dir)

    def _monitor_loop(self, process, msg_queue, cancel_event):
        cancel_deadline = None
        while process.is_alive():
            if self.future.cancelled_requested():
                if not cancel_event.is_set():
                    cancel_event.set()
                    cancel_deadline = time.monotonic() + _CANCEL_GRACE_SEC
                elif cancel_deadline is not None and time.monotonic() >= cancel_deadline:
                    logger.warning(f"Task cancel grace period elapsed, force terminating, func={self.func.__name__}")
                    break
            self._drain_queue(msg_queue)
            process.join(timeout=_POLL_INTERVAL)
        self._drain_queue(msg_queue)
        if not self.future.done:
            if self.future.cancelled_requested():
                self.future.cancel()
            elif process.exitcode != 0:
                logger.error(f"Subprocess exited with code {process.exitcode}, func={self.func.__name__}")
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
                logger.error(f"Subprocess error: {exc_name}: {exc_msg}, func={self.func.__name__}")
                self.future.set_exception(RuntimeError(f"{exc_name}: {exc_msg}"))

    def _cleanup(self, process, msg_queue, task_tmp_dir=None):
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
        if task_tmp_dir:
            shutil.rmtree(task_tmp_dir, ignore_errors=True)
