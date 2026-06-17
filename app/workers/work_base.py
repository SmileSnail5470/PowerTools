from app.ui.common.utils import get_file_type


def _worker_task_func(worker, *args, progress_cb=None, cancel_requested=None, **kwargs):
    if cancel_requested and cancel_requested():
        raise InterruptedError("Task was cancelled before start")

    result = worker.run_algorithm(progress_cb, cancel_requested, *args, **kwargs)

    if cancel_requested and cancel_requested():
        raise InterruptedError("Task was cancelled after execution")
    return result


class BaseWorker():
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def to_worker(self):
        """转换成 TaskManager 可用的 Worker
        """
        return _worker_task_func, (self, *self.args), self.kwargs
    
    def file_type(self, input_file):
        return get_file_type(input_file=input_file)