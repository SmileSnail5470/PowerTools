import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout, QLabel, QStackedWidget, QStackedLayout
from PySide6.QtGui import QFont, QColor

from app.ui.library.qfluentwidgets import PushButton, setFont, HeaderCardWidget, SegmentedWidget, ScrollArea

from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.file_selector_widget import FileSelectorWidget
from app.ui.widgets.directory_selector_widget import DirectorySelectorWidget
from app.ui.widgets.image_preview_widget import ImageNavigationWidget
from app.ui.widgets.status_bar_widget import StatusInfoWidget
from app.ui.widgets.ocr_preview_widget import OCRViewerWidget

from app.ui.common.event_bus import global_event_bus
from app.ui.common.task_params import bind_widget_to_param, TaskParams
from app.ui.common.task_status import TaskStatusModel
from app.ui.common.utils import get_file_type


ocr_params = TaskParams()
ocr_task_status_model = TaskStatusModel()


class FileSelectorCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("📁 文件选择"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)
        main_layout.addWidget(self.pivot, 0, Qt.AlignTop)
        main_layout.addWidget(self.stackedWidget)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        singleFileSelector = FileSelectorWidget(self)
        singleFileSelector.item_selected.connect(lambda file_path: global_event_bus.OCR_InputFileUpdate.emit(file_path))
        bind_widget_to_param(singleFileSelector, "item_selected", ocr_params, "input_path", transform=None)
        batchFilesSelector = DirectorySelectorWidget(self)
        batchFilesSelector.item_selected.connect(lambda file_path: global_event_bus.OCR_InputFileUpdate.emit(file_path))
        bind_widget_to_param(batchFilesSelector, "item_selected", ocr_params, "input_path", transform=None)

        self.addSubInterface(singleFileSelector, 'FileSelectorWidget', self.tr("文件"))
        self.addSubInterface(batchFilesSelector, 'DirectorySelectorWidget', self.tr("目录"))

        self.stackedWidget.setCurrentWidget(singleFileSelector)
        self.pivot.setCurrentItem(singleFileSelector.objectName())
        self.pivot.currentItemChanged.connect(
            lambda k:  self.stackedWidget.setCurrentWidget(self.findChild(QWidget, k)))

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(routeKey=objectName, text=text)


class ControlPanelWidget(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        view = QWidget(self)
        view.setObjectName('controlPanel')
        main_layout = QVBoxLayout(view)
        main_layout.setContentsMargins(0, 0, 12, 0)
        main_layout.setSpacing(10)
        main_layout.setAlignment(Qt.AlignTop)

        fileSelectorCard = FileSelectorCard(self)
        main_layout.addWidget(fileSelectorCard)

        main_layout.addStretch(1)

        self.setWidget(view)
        self.setViewportMargins(0, 0, 0, 0)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewWidget")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        self.stack = QStackedLayout()
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.placeholder_widget = QLabel("请选择图片文件进行 OCR 识别", parent=self)
        self.placeholder_widget.setStyleSheet("color: #888888;")  # 设置为浅灰色
        setFont(self.placeholder_widget, 20)
        self.placeholder_widget.setAlignment(Qt.AlignCenter)

        self.ocr_viewer = OCRViewerWidget(parent=self)

        self.stack.addWidget(self.placeholder_widget)
        self.stack.addWidget(self.ocr_viewer)
 

        main_layout.addLayout(self.stack, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        self.image_navigation_widget = ImageNavigationWidget(parent=self, task_type="ocr")
        bottom_layout.addWidget(self.image_navigation_widget, 3)

        # 底部状态栏
        status_info_widget = StatusInfoWidget(ocr_task_status_model, self)
        bottom_layout.addWidget(status_info_widget, 2)

        main_layout.addLayout(bottom_layout)

        self.files_preview_info = {}
        self.media_type = "image"

        global_event_bus.OCR_InputFileUpdate.connect(self.update_init_preview)
        global_event_bus.OCR_TaskFinished.connect(self.update_preview)
        global_event_bus.OCR_PreviewFile.connect(self._on_preview_file)
        global_event_bus.OCR_ImageNavigationInit.connect(lambda: self.image_navigation_widget.clear_images())

    def update_init_preview(self, file_path):
        self.image_navigation_widget.clear_images()
        self.ocr_viewer.init_scene()
        self.files_preview_info = {}
        self.media_type = "image"
        if not file_path:
            self.stack.setCurrentIndex(0)
            return
        if os.path.isdir(file_path):
            tmp_file_path = os.path.join(file_path, os.listdir(file_path)[0])
        else:
            tmp_file_path = file_path
        file_type = get_file_type(tmp_file_path)
        ext = tmp_file_path.lower().split(".")[-1]
        if file_type == "image":
            self.stack.setCurrentIndex(1)
            self.media_type = "image"
        else:
            self.placeholder_widget.setText(f"不支持的文件类型: {ext}")
            self.stack.setCurrentIndex(0)

    def update_preview(self, input_path, ocr_result):
        self.files_preview_info[input_path] = ocr_result
        self.image_navigation_widget.load_images([input_path], self.media_type)

    def _on_preview_file(self, path):
        out = self.files_preview_info.get(path)
        widget = self.stack.currentWidget()

        if self.media_type=="image" and out and widget and hasattr(widget, "set_data"):
            widget.set_data(image_path=path, raw_data=out)


class HeaderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        header = GradientHeader(parent=self, start=QColor(240, 147, 251), stop=QColor(245, 87, 108))
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)
        header_layout.setSpacing(10)

        title_label = QLabel(self.tr("📝 文字提取"))
        setFont(title_label, fontSize=24, weight=QFont.DemiBold)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
            }
        """)
        header_layout.addWidget(title_label)  
        header_layout.addStretch(1)

        self.process_btn = PushButton(text=self.tr("▶️ 开始处理"))
        self.process_btn.setStyleSheet("""
            PushButton {
                background-color: rgba(255, 255, 255, 0.55);
                color: #325c8a;
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                border: 1px solid rgba(255, 255, 255, 0.45);
            }
            PushButton:hover {
                background-color: rgba(255, 255, 255, 0.65);
            }
            PushButton:pressed {
                background-color: rgba(255, 255, 255, 0.50);
            }
        """)
        header_layout.addWidget(self.process_btn)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(header)

        self.process_btn.clicked.connect(self.ocr_process)

    def ocr_process(self):
        pass


class OCR(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("OCR")

        main_Layout = QVBoxLayout(self)
        main_Layout.setContentsMargins(0, 0, 0, 0)
        main_Layout.setSpacing(0)

        header = HeaderWidget(self)
        main_Layout.addWidget(header, 0, Qt.AlignTop)

        view_layout = QHBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # 左侧控制面板
        control_panel_widget = ControlPanelWidget(self)
        view_layout.addWidget(control_panel_widget, 3)

        # 右侧预览
        right_content = PreviewWidget(self)
        view_layout.addWidget(right_content, 7)

        main_Layout.addLayout(view_layout)