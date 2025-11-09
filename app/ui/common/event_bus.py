from PySide6.QtCore import QObject, Signal

class GlobalEventBus(QObject):
    # 水印添加 UI
    watermarkAdd_InputFileUpdate = Signal(str)
    watermarkAdd_TaskFinished = Signal(str, str)
    watermarkAdd_PreviewFile = Signal(str)


global_event_bus = GlobalEventBus()