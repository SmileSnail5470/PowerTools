from app.workers.work_base import BaseWorker


class WatermarkAddWork(BaseWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        if cancel_requested and cancel_requested():
            raise InterruptedError("WatermarkAdd task was cancelled before start")

        pass

        if cancel_requested and cancel_requested():
            raise InterruptedError("WatermarkAdd task was cancelled after execution")
        return 