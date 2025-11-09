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