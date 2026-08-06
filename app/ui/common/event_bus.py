from PySide6.QtCore import QObject, Signal

class GlobalEventBus(QObject):
    # 许可信息更新
    License_update = Signal()
    # 水印添加 UI
    watermarkAdd_InputFileUpdate = Signal(str)
    watermarkAdd_TaskFinished = Signal(str, str)
    watermarkAdd_PreviewFile = Signal(str)
    watermarkAdd_ImageNavigationInit = Signal()
    watermarkAdd_TaskFinishedByModel = Signal(str)

    # 水印移除 UI
    watermarkRemove_InputFileUpdate = Signal(str)
    watermarkRemove_TaskFinished = Signal(str, str)
    watermarkRemove_TaskProgress = Signal(str, str, str)
    watermarkRemove_PreviewFile = Signal(str)
    watermarkRemove_ImageNavigationInit = Signal()
    watermarkRemove_ManualMaskUpdate = Signal(str)
    watermarkRemove_TaskFinishedByModel = Signal(str)

    # OCR UI
    OCR_InputFileUpdate = Signal(str)
    OCR_TaskFinished = Signal(str, tuple)
    OCR_PreviewFile = Signal(str)
    OCR_ImageNavigationInit = Signal()
    OCR_TaskFinishedByModel = Signal(str)

    # 暗水印 UI
    blindWatermarkRemove_InputFileUpdate = Signal(str)
    blindWatermarkRemove_TaskFinished = Signal(str, tuple)
    blindWatermarkRemove_PreviewFile = Signal(str)
    blindWatermarkRemove_ImageNavigationInit = Signal()
    blindWatermarkRemove_TaskFinishedByModel = Signal(str)

    # 图像编辑 UI
    imageEdit_InputFileUpdate = Signal(str)
    imageEdit_TaskFinished = Signal(str, str)
    imageEdit_PreviewFile = Signal(str)
    imageEdit_ImageNavigationInit = Signal()
    imageEdit_TaskFinishedByModel = Signal(str)
    imageEdit_PreviewMode = Signal(str)


global_event_bus = GlobalEventBus()