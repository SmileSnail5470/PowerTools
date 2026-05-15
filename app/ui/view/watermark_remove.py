import os
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QWidget, QLabel, QHBoxLayout, QStackedWidget, QLineEdit, QFileDialog, QStackedLayout, 
    QSlider
)
from PySide6.QtGui import QFont, QColor, QAction

from app.ui.library.qfluentwidgets import( 
    setFont, HeaderCardWidget, SegmentedWidget, ScrollArea, PushButton, CaptionLabel,
    LineEdit, FluentIcon, ComboBox, TeachingTip, InfoBarIcon, TeachingTipTailPosition,
    MessageBox
)

from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.file_selector_widget import FileSelectorWidget
from app.ui.widgets.directory_selector_widget import DirectorySelectorWidget
from app.ui.widgets.custom_card_group_widget import StyleCard, CardSeparator
from app.ui.widgets.image_preview_widget import SyncImageViewer, ImageNavigationWidget
from app.ui.widgets.video_preview_widget import SyncVideoViewer
from app.ui.widgets.status_bar_widget import StatusInfoWidget
from app.ui.widgets.task_info_messagebox_widget import TaskInfoMessageBox
from app.ui.widgets.custom_navigation import CustomNavigation
from app.ui.widgets.watermark_detect_settings import WatermarkDetectSettings

from app.ui.common.event_bus import global_event_bus
from app.controllers.task_manager import global_task_manager
from app.ui.common.task_params import bind_widget_to_param, TaskParams
from app.ui.common.task_status import TaskStatusModel
from app.ui.common.utils import get_file_type
from app.ui.common.config import cfg

from app.workers.watermark_remove import WatermarkRemoveWork


watermark_remove_params = TaskParams()
watermark_remove_task_status_model = TaskStatusModel()

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
        singleFileSelector.item_selected.connect(lambda file_path: global_event_bus.watermarkRemove_InputFileUpdate.emit(file_path))
        bind_widget_to_param(singleFileSelector, "item_selected", watermark_remove_params, "input_path", transform=None)
        batchFilesSelector = DirectorySelectorWidget(self)
        batchFilesSelector.item_selected.connect(lambda file_path: global_event_bus.watermarkRemove_InputFileUpdate.emit(file_path))
        bind_widget_to_param(batchFilesSelector, "item_selected", watermark_remove_params, "input_path", transform=None)

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


class WatermarkDetectionTypeCard(HeaderCardWidget):
    detect_type = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🔍 水印检测方式"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        watermark_detect_settings = WatermarkDetectSettings(parent=self)
        bind_widget_to_param(watermark_detect_settings, "watermarkDetectType", watermark_remove_params, "watermark_detect_type", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkFormat", watermark_remove_params, "watermark_format", transform=None)
        bind_widget_to_param(watermark_detect_settings, "manualWatermarktMaskPath", watermark_remove_params, "manual_watermark_mask_path", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkAIInteractiveType", watermark_remove_params, "watermark_ai_interactive_type", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkDetectPrompt", watermark_remove_params, "watermark_detect_prompt", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkBoxes", watermark_remove_params, "watermark_boxes", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkContent", watermark_remove_params, "watermark_content", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkConfidence", watermark_remove_params, "watermark_confidence", transform=None)
        # 设置参数默认值
        watermark_detect_settings.watermarkDetectType.emit("ai_interactive_detect")
        watermark_detect_settings.watermarkFormat.emit("static_watermark")
        watermark_detect_settings.manualWatermarktMaskPath.emit("")
        watermark_detect_settings.watermarkAIInteractiveType.emit("semantic_detect")
        watermark_detect_settings.watermarkDetectPrompt.emit("")
        watermark_detect_settings.watermarkBoxes.emit([])
        watermark_detect_settings.watermarkContent.emit("general_watermark")
        watermark_detect_settings.watermarkConfidence.emit(0.5)

        global_event_bus.watermarkRemove_InputFileUpdate.connect(lambda file_path: watermark_detect_settings.set_file_path(file_path=file_path))

        main_layout.addWidget(watermark_detect_settings)


class WatermarkMaskDilate(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("⬆️ 水印 Mask"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(10)

        self.label = QLabel("2")
        self.label.setFixedWidth(25)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #666666;")

        # 滑块
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(8)
        self.slider.setValue(2)
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
        label = QLabel(self.tr("扩张系数"))
        label.setStyleSheet("color: #666666;")
        slider_layout.addWidget(label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.label)
        bind_widget_to_param(self.slider, "valueChanged", watermark_remove_params, "mask_dilate", transform=None)
        self.slider.valueChanged.connect(self.on_value_changed)
        self.slider.valueChanged.emit(2)
        main_layout.addLayout(slider_layout)

    def on_value_changed(self, value):
        self.label.setText(str(value))


class WatermarkRemoveStyleCard(HeaderCardWidget):
    model_name = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🎨 水印去除风格"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        self.cards: list[StyleCard] = []
        patchwiper_card = StyleCard("#4facfe", self.tr("细节增强"), self.tr("智能重建细节，提升清晰度，速度慢"))
        patchwiper_card.set_name("patchwiper")
        self.cards.append(patchwiper_card)
        main_layout.addWidget(patchwiper_card)
        main_layout.addWidget(CardSeparator(self))

        emdf_card = StyleCard("#f093fb", self.tr("智能修补"), self.tr("效果稳定，可能丢失细节，速度稍慢"))
        emdf_card.set_name("emdf")
        self.cards.append(emdf_card)
        main_layout.addWidget(emdf_card)
        main_layout.addWidget(CardSeparator(self))

        grig_card = StyleCard("#a18cd1", self.tr("平衡修复"), self.tr("结构保持好，可能丢失细节，速度快"))
        grig_card.set_name("grig")
        self.cards.append(grig_card)
        main_layout.addWidget(grig_card)
        main_layout.addWidget(CardSeparator(self))

        lama_card = StyleCard("#84fab0", self.tr("自然保守"), self.tr("画面衔接自然，可能丢失细节，速度适中"))
        lama_card.set_name("lama")
        self.cards.append(lama_card)
        main_layout.addWidget(lama_card)
        main_layout.addWidget(CardSeparator(self))

        coordfill_card = StyleCard("#fbc2eb", self.tr("快速填充"), self.tr("细节表现一般，适合简单背景，速度极快"))
        coordfill_card.set_name("coordfill")
        self.cards.append(coordfill_card)
        main_layout.addWidget(coordfill_card)
        main_layout.addWidget(CardSeparator(self))

        patchwiper_card.set_selected(True)
        for i, card in enumerate(self.cards):
            card.mousePressEvent = lambda event, c=card, idx=i: self.on_card_clicked(c, idx)

        bind_widget_to_param(self, "model_name", watermark_remove_params, "model_name", transform=None)
        self.model_name.emit("patchwiper")

        main_layout.addStretch()

    def on_card_clicked(self, card, index):
        # 取消所有卡片的选中状态
        for c in self.cards:
            c.set_selected(False)
        
        # 设置当前卡片为选中状态
        card.set_selected(True)
        self.model_name.emit(card.get_name())

    
class OutputSettingsCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("💾 输出设置"))
        self.setBorderRadius(8)
        self.viewLayout.setContentsMargins(10, 10, 10, 10)

        output_settings = QWidget()
        output_settings_layout = QVBoxLayout(output_settings)
        output_settings_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        output_settings_layout.setContentsMargins(0, 0, 0, 0)
        output_settings_layout.setSpacing(8)

        save_location_label = CaptionLabel(text=self.tr("保存位置"))
        setFont(save_location_label, 13)
        save_location_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        output_settings_layout.addWidget(save_location_label)
        self.save_location_line_edit = LineEdit()
        self.save_location_line_edit.setPlaceholderText(self.tr("选择保存位置"))
        save_location_action = QAction(FluentIcon.FOLDER_ADD.qicon(), "", triggered=self.save_location_browse)
        self.save_location_line_edit.addAction(save_location_action, QLineEdit.TrailingPosition)
        bind_widget_to_param(self.save_location_line_edit, "textChanged", watermark_remove_params, "output_path", transform=None)
        output_settings_layout.addWidget(self.save_location_line_edit)
        output_settings_layout.addSpacing(10)

        output_format_label = CaptionLabel(text=self.tr("输出格式"))
        setFont(output_format_label, 13)
        output_format_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        output_settings_layout.addWidget(output_format_label)
        self.output_format_combo = ComboBox()
        self.output_format_combo.addItems([self.tr("保持原格式")])
        bind_widget_to_param(self.output_format_combo, "currentTextChanged", watermark_remove_params, "output_format", transform=None)
        output_settings_layout.addWidget(self.output_format_combo)

        self.viewLayout.addWidget(output_settings)

        global_event_bus.watermarkRemove_InputFileUpdate.connect(self.update_output_format_combox)

    def update_output_format_combox(self, file_path):
        self.output_format_combo.clear()
        if not file_path:
            self.output_format_combo.addItems([self.tr("保持原格式")])
            return
        if os.path.isdir(file_path):
            file_type = get_file_type(os.path.join(file_path, os.listdir(file_path)[0]))
        else:
            file_type = get_file_type(file_path)
        if file_type == "image":
            self.output_format_combo.addItems([self.tr("保持原格式"), "JPG", "PNG"])
        else:
            self.output_format_combo.addItems([self.tr("保持原格式")])

    def save_location_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog
        )
        if directory:
            self.save_location_line_edit.setText(directory)

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

        watermarkDetectionTypeCard = WatermarkDetectionTypeCard(self)
        main_layout.addWidget(watermarkDetectionTypeCard)

        watermarkMaskDilate = WatermarkMaskDilate(self)
        main_layout.addWidget(watermarkMaskDilate)

        watermarkRemoveStyleCard = WatermarkRemoveStyleCard(self)
        main_layout.addWidget(watermarkRemoveStyleCard)

        outputSettingsCard = OutputSettingsCard(self)
        main_layout.addWidget(outputSettingsCard)

        self.setWidget(view)
        self.setViewportMargins(0, 0, 0, 0)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_layout.addStretch(1)


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewWidget")
        self.preview_mode = "ResultCompared"

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        # 添加导航栏
        self.navigation = CustomNavigation(navigation_item_texts=[self.tr("结果对比预览"), self.tr("Mask对比预览")], parent=self)
        self.navigation.currentTextChanged.connect(self._on_navigation_changed)
        main_layout.addWidget(self.navigation)
        self.navigation.hide()

        # 预览区域
        self.stack = QStackedLayout()
        self.stack.setObjectName("CompareStackWidget")
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.placeholder_widget = QLabel("请选择图片或视频文件进行预览", parent=self)
        self.placeholder_widget.setStyleSheet("color: #888888;")  # 设置为浅灰色
        setFont(self.placeholder_widget, 20)
        self.placeholder_widget.setAlignment(Qt.AlignCenter)

        self.image_viewer = SyncImageViewer(img1="", img2="", parent=self)
        self.video_viewer = SyncVideoViewer(self)

        self.stack.addWidget(self.placeholder_widget)
        self.stack.addWidget(self.image_viewer)
        self.stack.addWidget(self.video_viewer) 

        main_layout.addLayout(self.stack, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        self.image_navigation_widget = ImageNavigationWidget(parent=self, task_type="watermark_remove")
        bottom_layout.addWidget(self.image_navigation_widget, 3)

        # 底部状态栏
        status_info_widget = StatusInfoWidget(watermark_remove_task_status_model, self)
        bottom_layout.addWidget(status_info_widget, 2)

        main_layout.addLayout(bottom_layout)

        self.files_preview_info = {}
        self.media_type = "image"

        global_event_bus.watermarkRemove_InputFileUpdate.connect(self.update_init_preview)
        global_event_bus.watermarkRemove_TaskFinished.connect(self.update_preview)
        global_event_bus.watermarkRemove_TaskProgress.connect(self.update_process)
        global_event_bus.watermarkRemove_PreviewFile.connect(self._on_preview_file)
        global_event_bus.watermarkRemove_ImageNavigationInit.connect(lambda: self.image_navigation_widget.clear_images())

    def update_init_preview(self, file_path):
        self.image_navigation_widget.clear_images()
        self.image_viewer.init_scene()
        self.video_viewer.init_scene()
        self.files_preview_info = {}
        self.media_type = "image"
        if not file_path:
            self.navigation.hide()
            self.stack.setCurrentIndex(0)
            return
        self.navigation.show()
        if os.path.isdir(file_path):
            tmp_file_path = os.path.join(file_path, os.listdir(file_path)[0])
        else:
            tmp_file_path = file_path
        file_type = get_file_type(tmp_file_path)
        ext = tmp_file_path.lower().split(".")[-1]
        if file_type == "image":
            self.stack.setCurrentIndex(1)
            self.media_type = "image"
        elif file_type == "video":
            self.stack.setCurrentIndex(2)
            self.media_type = "video"
        else:
            self.navigation.hide()
            self.placeholder_widget.setText(f"不支持的文件类型: {ext}")
            self.stack.setCurrentIndex(0)

    def update_preview(self, input_path, output_path):
        preview_mode = "ResultCompared"
        if preview_mode not in self.files_preview_info:
            self.files_preview_info[preview_mode] = {}
        self.files_preview_info[preview_mode][input_path] = output_path
        current_file_path = self.image_navigation_widget.get_current_image()
        if current_file_path:
            self._on_preview_file(current_file_path)

    def update_process(self, input_path, value, mask_path):
        if value not in self.files_preview_info:
            self.files_preview_info[value] = {}
        self.files_preview_info[value][input_path] = mask_path
        self.image_navigation_widget.load_images([input_path], self.media_type)

    def _on_navigation_changed(self, text):
        if text == self.tr("结果对比预览"):
            self.preview_mode = "ResultCompared"
        elif text == self.tr("Mask对比预览"):
            self.preview_mode = "MaskCompleted"
        else:
            self.preview_mode = "ResultCompared"
        current_file_path = self.image_navigation_widget.get_current_image()
        if current_file_path:
            self._on_preview_file(current_file_path)

    def _on_preview_file(self, path):
        out = self.files_preview_info.get(self.preview_mode, {}).get(path, None)
        if out is None:
            if self.media_type == "image":
                self.image_viewer.init_scene()
                self.stack.setCurrentIndex(1)
            else:
                self.video_viewer.init_scene()
                self.stack.setCurrentIndex(2)
            return

        widget = self.stack.currentWidget()
        if self.media_type=="image" and out and widget and hasattr(widget, "set_images"):
            widget.set_images(img1=path, img2=out)
        elif self.media_type=="video" and out and widget and hasattr(widget, "setVideos"):
            widget.setVideos(main_path=path, sub_path=out)


class HeaderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        header = GradientHeader(parent=self, start=QColor(79, 172, 254), stop=QColor(0, 242, 254))
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)
        header_layout.setSpacing(10)

        title_label = QLabel(self.tr("🧹 水印移除工具"))
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

        self.process_btn.clicked.connect(self.remove_watermark_process)

    def remove_watermark_process(self):
        init_params = watermark_remove_params.to_dict()
        error_msg, task_params = self._params_check(params=init_params)
        if error_msg:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        w = TaskInfoMessageBox(task_params, "watermark-remove", self.window())
        if not w.exec():
            return
        total_tasks = []
        input_path = task_params["input_path"]
        watermark_remove_task_status_model.reset()
        global_event_bus.watermarkRemove_ImageNavigationInit.emit()
        if os.path.isdir(input_path):
            for one_file in os.listdir(input_path):
                task_params["input_path"] = os.path.join(input_path, one_file)
                task_instance = WatermarkRemoveWork(**task_params)
                func, args, kwargs = task_instance.to_worker()
                total_tasks.append((func, args, kwargs))
        else:
            task_instance = WatermarkRemoveWork(**task_params)
            func, args, kwargs = task_instance.to_worker()
            total_tasks.append((func, args, kwargs))

        watermark_remove_task_status_model.set_total(len(total_tasks))

        for func, args, kwargs in total_tasks:
            input_path = kwargs["input_path"]
            future = global_task_manager.submit(func, *args, **kwargs)
            
            future.finished.connect(
                lambda result, path=input_path: self._task_finished(path, result)
            )
            future.failed.connect(
                lambda e, path=input_path: watermark_remove_task_status_model.report_failure(path, e)
            )
            future.cancelled.connect(
                lambda path=input_path: watermark_remove_task_status_model.report_failure(path, "任务被取消")
            )
            future.progress.connect(
                lambda value, msg, path=input_path: self._task_progress(path, value, msg)
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

    def _task_progress(self, input_path, value, msg):
        global_event_bus.watermarkRemove_TaskProgress.emit(input_path, value, msg)

    def _task_finished(self, input_path, output_path):
        watermark_remove_task_status_model.report_success()
        global_event_bus.watermarkRemove_TaskFinished.emit(input_path, output_path)

    def _params_check(self, params):
        error_msg = ""
        task_params = {}
        if not params:
            error_msg = self.tr("请设置水印移除参数")
            return error_msg, task_params
        if not cfg.get(cfg.localWatermarkRemovalEnabled):
            error_msg = self.tr("请在设置页面打开 '水印去除AI能力' 开关")
            return error_msg, task_params
        
        if "input_path" not in params or not params["input_path"] or " " in params["input_path"]:
            error_msg = self.tr("请选择要处理的文件或目录且文件名不能有空格")
            return error_msg, task_params
        else:
            task_params["input_path"] = params["input_path"]

        if get_file_type(params["input_path"]) == "video" and cfg.get(cfg.additionalParams)["SoftwareSettings"]["FFmpeg_status_info"]["text"] != "OK":
            error_msg = self.tr("请在设置页面配置软件 FFmpeg 的正确路径并通过验证")
            return error_msg, task_params

        if "watermark_detect_type" not in params:
            error_msg = self.tr("请选择水印检测方式")
            return error_msg, task_params
        else:
            task_params["watermark_detect_type"] = params["watermark_detect_type"]

        if params["watermark_detect_type"] == "ai_interactive_detect" and not cfg.get(cfg.localObjectSegmentationEnabled):
            error_msg = self.tr("请在设置页面打开 '物体分割AI能力' 开关")
            return error_msg, task_params
        
        if "mask_dilate" not in params:
            error_msg = self.tr("请设置水印 Mask 扩张系数")
            return error_msg, task_params
        else:
            task_params["mask_dilate"] = params["mask_dilate"]

        if "model_name" not in params or not params["model_name"]:
            error_msg = self.tr("请选择水印移除算法")
            return error_msg, task_params
        else:
            task_params["model_name"] = params["model_name"]

        if "output_path" not in params or not params["output_path"]:
            error_msg = self.tr("请设置文件保存位置")
            return error_msg, task_params
        else:
            task_params["output_path"] = params["output_path"]
            task_params["output_format"] = params["output_format"]

        if params["watermark_detect_type"] == "ai_auto_detect":
            task_params["watermark_content"] = params["watermark_content"]
            task_params["watermark_format"] = params["watermark_format"]
        if params["watermark_detect_type"] == "ai_interactive_detect":
            task_params["watermark_ai_interactive_type"] = params["watermark_ai_interactive_type"]
            task_params["watermark_format"] = params["watermark_format"]
            task_params["watermark_confidence"] = params["watermark_confidence"]
            if task_params["watermark_ai_interactive_type"] == "semantic_detect":
                task_params["watermark_detect_prompt"] = params["watermark_detect_prompt"]
            if task_params["watermark_ai_interactive_type"] == "space_detect":
                task_params["watermark_boxes"] = params["watermark_boxes"]
        if params["watermark_detect_type"] == "manual_detect":
            task_params["manual_watermark_mask_path"] = params["manual_watermark_mask_path"]
        return error_msg, task_params


class WatermarkRemove(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WatermarkRemove")

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
        