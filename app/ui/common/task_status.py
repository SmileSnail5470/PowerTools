import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List
from PySide6.QtCore import QObject, Signal


class TaskStatusModel(QObject):
    updated = Signal(dict)

    def __init__(self):
        super().__init__()
        self.data = {
            "total": 0,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "failures": []
        }

    def reset(self):
        self.data = {k: 0 for k in ["total", "processed", "success", "failed"]}
        self.data["failures"] = []
        self.updated.emit(self.data.copy())

    def set_total(self, n: int):
        self.data["total"] = n
        self.updated.emit(self.data.copy())

    def report_success(self):
        self.data["processed"] += 1
        self.data["success"] += 1
        self.updated.emit(self.data.copy())

    def report_failure(self, filename, reason):
        self.data["processed"] += 1
        self.data["failed"] += 1
        self.data["failures"].append((filename, reason))
        self.updated.emit(self.data.copy())


class TaskState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    FAILED = "failed"


class StepState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass
class PipelineStep:
    name: str
    state: StepState = StepState.PENDING
    start_time: float = 0.0
    duration: float = 0.0


@dataclass
class ActiveWorker:
    worker_id: str
    filename: str
    step_name: str
    progress: float = 0.0

@dataclass
class BatchStatus:
    total: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    failures: list = field(default_factory=list)


@dataclass
class PerformanceStatus:
    start_time: float = 0.0
    elapsed_seconds: float = 0.0
    avg_seconds_per_file: float = 0.0
    current_seconds_per_file: float = 0.0
    eta_seconds: float = 0.0


@dataclass
class BackendStatus:
    backend_type: str = "CPU"
    gpu_name: str = ""
    worker_count: int = 0


@dataclass
class TaskStatus:
    state: TaskState = TaskState.IDLE
    batch: BatchStatus = field(default_factory=BatchStatus)
    performance: PerformanceStatus = field(default_factory=PerformanceStatus)
    backend: BackendStatus = field(default_factory=BackendStatus)
    pipeline_steps: List[PipelineStep] = field(default_factory=list)
    active_workers: List[ActiveWorker] = field(default_factory=list)


class TmpTaskStatusModel(QObject):
    changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.status = TaskStatus()

    def notify(self):
        self.changed.emit(self.status)

    def reset(self):
        self.status = TaskStatus()
        self.notify()

    def start_batch(self, total: int, backend_type: str = "CPU 运行", worker_count: int = 1, gpu_name: str = ""):
        self.reset()
        self.status.state = TaskState.RUNNING
        self.status.batch.total = total
        self.status.performance.start_time = time.time()
        self.status.backend.backend_type = backend_type
        self.status.backend.worker_count = worker_count
        self.status.backend.gpu_name = gpu_name
        self.notify()

    def set_pipeline_steps(self, names: list[str]):
        self.status.pipeline_steps = [PipelineStep(name=n) for n in names]
        self.notify()

    def start_step(self, name: str):
        for step in self.status.pipeline_steps:
            if step.name == name:
                step.state = StepState.RUNNING
                step.start_time = time.time()
                break
        self.notify()

    def finish_step(self, name: str):
        for step in self.status.pipeline_steps:
            if step.name == name:
                step.state = StepState.COMPLETED
                step.duration = (time.time() - step.start_time)
                break
        self.notify()

    def update_worker(self, worker_id: str, filename: str, step_name: str, progress: float):
        for worker in self.status.active_workers:
            if worker.worker_id == worker_id:
                worker.filename = filename
                worker.step_name = step_name
                worker.progress = progress
                self.notify()
                return
        self.status.active_workers.append(ActiveWorker(worker_id=worker_id, filename=filename, step_name=step_name, progress=progress))
        self.notify()

    def remove_worker(self, worker_id: str):
        self.status.active_workers = [w for w in self.status.active_workers if w.worker_id != worker_id]
        self.notify()

    def report_success(self):
        self.status.batch.processed += 1
        self.status.batch.success += 1
        self._update_performance()
        self._check_finished()
        self.notify()

    def report_failure(self, filename: str, reason: str):
        self.status.batch.processed += 1
        self.status.batch.failed += 1
        self.status.batch.failures.append((filename, reason))
        self._update_performance()
        self._check_finished()
        self.notify()

    def _update_performance(self):
        perf = self.status.performance
        processed = self.status.batch.processed
        if processed == 0:
            return
        elapsed = (time.time() - perf.start_time)
        avg = elapsed / processed
        remain = self.status.batch.total - processed
        perf.elapsed_seconds = elapsed
        perf.avg_seconds_per_file = avg
        perf.current_seconds_per_file = avg
        perf.eta_seconds = remain * avg

    def _check_finished(self):
        batch = self.status.batch
        if batch.total > 0 and batch.processed >= batch.total:
            self.status.state = TaskState.FINISHED
            self.status.performance.eta_seconds = 0

    @property
    def progress_percent(self) -> float:
        total = self.status.batch.total
        if total == 0:
            return 0
        return (self.status.batch.processed / total) * 100