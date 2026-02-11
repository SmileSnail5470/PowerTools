from PySide6.QtCore import QObject, Signal

class GlobalEventBus(QObject):
    # 水印添加 UI
    watermarkAdd_InputFileUpdate = Signal(str)
    watermarkAdd_TaskFinished = Signal(str, str)
    watermarkAdd_PreviewFile = Signal(str)
    watermarkAdd_ImageNavigationInit = Signal()

    # 水印移除 UI
    watermarkRemove_InputFileUpdate = Signal(str)
    watermarkRemove_TaskFinished = Signal(str, str)
    watermarkRemove_PreviewFile = Signal(str)
    watermarkRemove_ImageNavigationInit = Signal()
    watermarkRemove_ManualMaskUpdate = Signal(str)

    # OCR UI
    OCR_InputFileUpdate = Signal(str)
    OCR_TaskFinished = Signal(str, tuple)
    OCR_PreviewFile = Signal(str)
    OCR_ImageNavigationInit = Signal()


global_event_bus = GlobalEventBus()