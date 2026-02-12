import os
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout, QLabel, QStackedWidget, QStackedLayout, QSlider
from PySide6.QtGui import QFont, QColor

from app.ui.library.qfluentwidgets import (
    PushButton, setFont, HeaderCardWidget, SegmentedWidget, ScrollArea, ComboBox, BodyLabel,
    TeachingTip, InfoBarIcon, TeachingTipTailPosition, MessageBox
)

from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.file_selector_widget import FileSelectorWidget
from app.ui.widgets.directory_selector_widget import DirectorySelectorWidget
from app.ui.widgets.image_preview_widget import ImageNavigationWidget
from app.ui.widgets.status_bar_widget import StatusInfoWidget
from app.ui.widgets.ocr_preview_widget import OCRViewerWidget
from app.ui.widgets.toggle_switch_widget import ToggleSwitch
from app.ui.widgets.task_info_messagebox_widget import TaskInfoMessageBox
from app.ui.widgets.custom_card_group_widget import CustomCardGroupWidget, StyleCard, CardSeparator

from app.ui.common.event_bus import global_event_bus
from app.controllers.task_manager import global_task_manager
from app.ui.common.task_params import bind_widget_to_param, TaskParams
from app.ui.common.task_status import TaskStatusModel
from app.ui.common.utils import get_file_type
from app.ui.common.config import cfg

from app.workers.ocr_work import OCRWork


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


class ModelStyleCard(HeaderCardWidget):
    model_name = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🎨 OCR 模型类型"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        self.cards: list[StyleCard] = []
        pp_ocr_card = StyleCard("#4facfe", self.tr("通用识别"), self.tr("用于通用场景文字识别，不适合复杂图表公式等识别"))
        pp_ocr_card.set_name("pp_ocr")
        self.cards.append(pp_ocr_card)
        main_layout.addWidget(pp_ocr_card)
        main_layout.addWidget(CardSeparator(self))

        pp_ocr_card.set_selected(True)
        for i, card in enumerate(self.cards):
            card.mousePressEvent = lambda event, c=card, idx=i: self.on_card_clicked(c, idx)

        bind_widget_to_param(self, "model_name", ocr_params, "model_name", transform=None)
        self.model_name.emit("pp_ocr")

        main_layout.addStretch()

    def on_card_clicked(self, card, index):
        # 取消所有卡片的选中状态
        for c in self.cards:
            c.set_selected(False)
        
        # 设置当前卡片为选中状态
        card.set_selected(True)
        self.model_name.emit(card.get_name())


class SettingsCard(HeaderCardWidget):
    language_map = {
        "中日英": "zh-jp-en"
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("⚙️ OCR 设置"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        language_content_combox = ComboBox()
        setFont(language_content_combox, 12)
        language_content_combox.addItems(["中日英"])
        watermark_content_card = CustomCardGroupWidget(
            title=self.tr("识别语言"), 
            content=self.tr("指定要识别的目标语言"), 
            parent=self,
            text_layout_contents_margins=(12, 0, 0, 0),
            label_v_space=2
        )
        watermark_content_card.addWidget(language_content_combox, stretch=0)
        watermark_content_card.setSeparatorVisible(True)
        bind_widget_to_param(language_content_combox, "currentTextChanged", ocr_params, "ocr_rec_language", transform=None)
        language_content_combox.currentTextChanged.emit(self.language_map["中日英"])
        main_layout.addWidget(watermark_content_card)

        self.toggle_switch_btns: list[ToggleSwitch] = []
        use_textline_ori_btn= ToggleSwitch(on_color="#4FACFE")
        use_textline_ori_btn.setActive(False)
        self.toggle_switch_btns.append(use_textline_ori_btn)
        use_textline_ori_card = CustomCardGroupWidget(
            title=self.tr("文本行方向矫正"), 
            content=self.tr("开启后，可以自动识别和矫正 0° 和 180° 的文本行"), 
            parent=self,
            text_layout_contents_margins=(12, 0, 0, 0),
            label_v_space=2
        )
        use_textline_ori_card.setSeparatorVisible(True)
        use_textline_ori_card.setContentWordWrap(True)
        use_textline_ori_card.addWidget(use_textline_ori_btn)
        main_layout.addWidget(use_textline_ori_card)
        bind_widget_to_param(use_textline_ori_btn, "toggled", ocr_params, "use_textline_ori", transform=None)
        use_textline_ori_btn.toggled.emit(False)

        slider_main_layout = QVBoxLayout()
        slider_main_layout.setContentsMargins(12, 10, 24, 10)
        slider_main_layout.setSpacing(10)
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addSpacing(10)
        self.label = QLabel("0.5")
        self.label.setFixedWidth(25)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #666666;")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(10)
        self.slider.setValue(5)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setSingleStep(1)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #ddd;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #2196F3;
                border-radius: 3px;
            }
        """)
        label = BodyLabel(self.tr("识别置信度"))
        slider_main_layout.addWidget(label)
        slider_main_layout.addLayout(slider_layout)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.label)
        bind_widget_to_param(self.slider, "valueChanged", ocr_params, "drop_score", transform=lambda x: x / 10)
        self.slider.valueChanged.connect(self.on_value_changed)
        self.slider.valueChanged.emit(5)
        main_layout.addLayout(slider_main_layout)

    def on_value_changed(self, value):
        self.label.setText(str(value / 10))


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

        modelStyleCard = ModelStyleCard(self)
        main_layout.addWidget(modelStyleCard)

        settingsCard = SettingsCard(self)
        main_layout.addWidget(settingsCard)


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
        init_params = ocr_params.to_dict()
        error_msg, task_params = self._params_check(params=init_params)
        if error_msg:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        w = TaskInfoMessageBox(task_params, "ocr-rec", self.window())
        if not w.exec():
            return
        total_tasks = []
        input_path = task_params["input_path"]
        ocr_task_status_model.reset()
        global_event_bus.OCR_ImageNavigationInit.emit()
        if os.path.isdir(input_path):
            for one_file in os.listdir(input_path):
                task_params["input_path"] = os.path.join(input_path, one_file)
                task_instance = OCRWork(**task_params)
                func, args, kwargs = task_instance.to_worker()
                total_tasks.append((func, args, kwargs))
        else:
            task_instance = OCRWork(**task_params)
            func, args, kwargs = task_instance.to_worker()
            total_tasks.append((func, args, kwargs))

        ocr_task_status_model.set_total(len(total_tasks))

        for func, args, kwargs in total_tasks:
            input_path = kwargs["input_path"]
            future = global_task_manager.submit(func, *args, **kwargs)
            
            future.finished.connect(
                lambda result, path=input_path: self._task_finished(path, result)
            )
            future.failed.connect(
                lambda e, path=input_path: ocr_task_status_model.report_failure(path, e)
            )
            future.cancelled.connect(
                lambda path=input_path: ocr_task_status_model.report_failure(path, "任务被取消")
            )

        TeachingTip.create(
            target=self.process_btn,
            icon=InfoBarIcon.SUCCESS,
            title=self.tr("通知"),
            content=self.tr("任务提交成功"),
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2000,
            parent=self
        )

    def _task_finished(self, input_path, ocr_result):
        ocr_task_status_model.report_success()
        global_event_bus.OCR_TaskFinished.emit(input_path, ocr_result)

    def _params_check(self, params):
        error_msg = ""
        task_params = {}
        if not params:
            error_msg = self.tr("请设置OCR识别参数")
            return error_msg, task_params
        if not cfg.get(cfg.localOCREnabled):
            error_msg = self.tr("请在设置页面打开 'OCR 能力' 开关")
            return error_msg, task_params
        
        if "input_path" not in params or not params["input_path"] or " " in params["input_path"]:
            error_msg = self.tr("请选择要处理的文件或目录且文件名不能有空格")
            return error_msg, task_params
        else:
            task_params["input_path"] = params["input_path"]

        if "model_name" not in params or not params["model_name"]:
            error_msg = self.tr("请选择OCR识别算法")
            return error_msg, task_params
        else:
            task_params["model_name"] = params["model_name"]

        if "ocr_rec_language" not in params or not params["ocr_rec_language"]:
            error_msg = self.tr("请设置要识别的目标语言")
            return error_msg, task_params
        else:
            task_params["ocr_rec_language"] = params["ocr_rec_language"]

        if "drop_score" not in params or not params["drop_score"]:
            error_msg = self.tr("请设置识别置信度")
            return error_msg, task_params
        else:
            task_params["drop_score"] = params["drop_score"]

        if "use_textline_ori" in params:
            task_params["use_textline_ori"] = params["use_textline_ori"]

        return error_msg, task_params


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