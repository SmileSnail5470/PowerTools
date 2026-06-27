import os
import sys
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QWidget, QLabel, QHBoxLayout, QStackedWidget, QStackedLayout, QLineEdit, QFileDialog,
    QSlider, QButtonGroup, QPushButton
)
from PySide6.QtGui import QFont, QColor, QAction

from app.ui.library.qfluentwidgets import (
    setFont, HeaderCardWidget, SegmentedWidget, ScrollArea, PushButton, CaptionLabel,
    LineEdit, FluentIcon, ComboBox, TeachingTip, InfoBarIcon, TeachingTipTailPosition,
    MessageBox
)

from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.file_selector_widget import FileSelectorWidget
from app.ui.widgets.directory_selector_widget import DirectorySelectorWidget
from app.ui.widgets.custom_card_group_widget import StyleCard, CardSeparator
from app.ui.widgets.image_preview_widget import SyncImageViewer, ImageNavigationWidget
from app.ui.widgets.status_bar_widget import StatusInfoWidget

from app.ui.common.event_bus import global_event_bus
from app.ui.common.task_params import bind_widget_to_param, TaskParams
from app.ui.common.task_status import TaskStatusModel
from app.ui.common.utils import get_file_type
from app.ui.common.config import cfg


blind_watermark_remove_params = TaskParams()
blind_watermark_remove_task_status_model = TaskStatusModel()


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
        singleFileSelector.item_selected.connect(lambda file_path: global_event_bus.blindWatermarkRemove_InputFileUpdate.emit(file_path))
        bind_widget_to_param(singleFileSelector, "item_selected", blind_watermark_remove_params, "input_path", transform=None)
        batchFilesSelector = DirectorySelectorWidget(self)
        batchFilesSelector.item_selected.connect(lambda file_path: global_event_bus.blindWatermarkRemove_InputFileUpdate.emit(file_path))
        bind_widget_to_param(batchFilesSelector, "item_selected", blind_watermark_remove_params, "input_path", transform=None)

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


class CropSizeCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("✂️ 图像裁剪大小"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(10)

        self.label = QLabel("128")
        self.label.setFixedWidth(30)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #666666;")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(64)
        self.slider.setMaximum(256)
        self.slider.setValue(128)
        self.slider.setSingleStep(32)
        self.slider.setTickInterval(32)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
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

        label = QLabel(self.tr("裁剪尺寸 (px)"))
        label.setStyleSheet("color: #666666;")
        slider_layout.addWidget(label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.label)

        bind_widget_to_param(self.slider, "valueChanged", blind_watermark_remove_params, "crop_size", transform=None)
        self.slider.valueChanged.connect(self.on_value_changed)
        self.slider.valueChanged.emit(128)

        main_layout.addLayout(slider_layout)

        hint_label = CaptionLabel(self.tr("建议值：128。值越大内存/显存占用越高"))
        hint_label.setStyleSheet("color: #999999;")
        main_layout.addWidget(hint_label)

    def on_value_changed(self, value):
        self.label.setText(str(value))


class ModelSelectCard(HeaderCardWidget):
    model_name = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🤖 去除模型"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        tab_widget = QWidget()
        tab_layout = QHBoxLayout(tab_widget)
        tab_layout.setContentsMargins(4, 4, 4, 4)
        tab_layout.setSpacing(4)
        tab_widget.setStyleSheet("""
            QWidget {
                background-color: #f1f5f9;
                border-radius: 8px;
            }
            QPushButton {
                background-color: transparent;
                color: #7f8c8d;
                padding: 6px 0;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                color: #2c3e50;
                background-color: rgba(255, 255, 255, 0.4);
            }
            QPushButton:checked {
                background-color: #ffffff;
                color: #3498db;
                border: 1px solid rgba(0, 0, 0, 0.03);
            }
        """)
        self.tab_group = QButtonGroup(self)
        self.tab_group.setExclusive(True)
        self.tab_image = QPushButton(self.tr("图片模型"))
        setFont(self.tab_image, fontSize=14, weight=QFont.DemiBold)
        self.tab_image.setCheckable(True)
        self.tab_image.setChecked(True)
        self.tab_video = QPushButton(self.tr("视频模型"))
        setFont(self.tab_video, fontSize=14, weight=QFont.DemiBold)
        self.tab_video.setCheckable(True)
        self.tab_video.setEnabled(False)  # 暂不支持视频模型
        self.tab_group.addButton(self.tab_image, 0)
        self.tab_group.addButton(self.tab_video, 1)
        tab_layout.addWidget(self.tab_image)
        tab_layout.addWidget(self.tab_video)
        main_layout.addWidget(tab_widget)

        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 6, 0, 6)
        image_layout.setSpacing(0)

        blind_remove_card = StyleCard("#8b5cf6", self.tr("盲水印去除"), self.tr("基于深度学习的暗水印去除"))
        blind_remove_card.set_name("blind_watermark_removal")
        blind_remove_card.set_selected(True)
        image_layout.addWidget(blind_remove_card)
        image_layout.addStretch()
        self.stacked_widget.addWidget(image_container)

        self.image_cards = [blind_remove_card]

        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 6, 0, 6)
        video_layout.setSpacing(0)

        video_placeholder = QLabel(self.tr("视频盲水印去除模型开发中，敬请期待..."))
        video_placeholder.setStyleSheet("color: #999999; padding: 20px;")
        video_placeholder.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(video_placeholder)
        video_layout.addStretch()
        self.stacked_widget.addWidget(video_container)

        self.tab_image.toggled.connect(lambda checked: self._on_tab_changed(0, checked))

        bind_widget_to_param(self, "model_name", blind_watermark_remove_params, "model_name", transform=None)
        self.model_name.emit("blind_watermark_removal")

    def _on_tab_changed(self, index, checked):
        if checked:
            self.stacked_widget.setCurrentIndex(index)


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
        save_location_label.setStyleSheet("color: #888888;")
        output_settings_layout.addWidget(save_location_label)
        self.save_location_line_edit = LineEdit()
        self.save_location_line_edit.setPlaceholderText(self.tr("选择保存位置"))
        save_location_action = QAction(FluentIcon.FOLDER_ADD.qicon(), "", triggered=self.save_location_browse)
        self.save_location_line_edit.addAction(save_location_action, QLineEdit.TrailingPosition)
        bind_widget_to_param(self.save_location_line_edit, "textChanged", blind_watermark_remove_params, "output_path", transform=None)
        output_settings_layout.addWidget(self.save_location_line_edit)

        self.viewLayout.addWidget(output_settings)

    def save_location_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择文件夹", "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0)
        )
        if directory:
            self.save_location_line_edit.setText(directory)


class ControlPanelWidget(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        view = QWidget(self)
        view.setObjectName('blindWatermarkRemoveControlPanel')
        main_layout = QVBoxLayout(view)
        main_layout.setContentsMargins(0, 0, 12, 0)
        main_layout.setSpacing(10)
        main_layout.setAlignment(Qt.AlignTop)

        main_layout.addWidget(FileSelectorCard(self))
        main_layout.addWidget(CropSizeCard(self))
        main_layout.addWidget(ModelSelectCard(self))
        main_layout.addWidget(OutputSettingsCard(self))

        self.setWidget(view)
        self.setViewportMargins(0, 0, 0, 0)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_layout.addStretch(1)


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BlindWatermarkRemovePreview")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        self.stack = QStackedLayout()
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.placeholder_widget = QLabel(self.tr("请选择图片文件进行盲水印去除预览"), parent=self)
        self.placeholder_widget.setStyleSheet("color: #888888;")
        setFont(self.placeholder_widget, 20)
        self.placeholder_widget.setAlignment(Qt.AlignCenter)

        self.image_viewer = SyncImageViewer(img1="", img2="", parent=self)

        self.stack.addWidget(self.placeholder_widget)
        self.stack.addWidget(self.image_viewer)

        main_layout.addLayout(self.stack, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        self.image_navigation_widget = ImageNavigationWidget(parent=self, task_type="blind_watermark_remove")
        bottom_layout.addWidget(self.image_navigation_widget, 3)

        self.status_info_widget = StatusInfoWidget(blind_watermark_remove_task_status_model, self)
        self.status_info_widget.model.set_pipeline_steps(names=[self.tr('准备任务'), self.tr('盲水印去除'), self.tr('导出结果')])
        bottom_layout.addWidget(self.status_info_widget, 2)

        main_layout.addLayout(bottom_layout)

        self.files_preview_info = {}
        self.media_type = "image"

        global_event_bus.blindWatermarkRemove_InputFileUpdate.connect(self.update_init_preview)
        global_event_bus.blindWatermarkRemove_TaskFinished.connect(self.update_preview)
        global_event_bus.blindWatermarkRemove_PreviewFile.connect(self._on_preview_file)
        global_event_bus.blindWatermarkRemove_ImageNavigationInit.connect(lambda: self.image_navigation_widget.clear_images())

    def update_init_preview(self, file_path):
        self.image_navigation_widget.clear_images()
        self.image_viewer.init_scene()
        self.files_preview_info = {}
        self.media_type = "image"
        if not file_path:
            self.stack.setCurrentIndex(0)
            blind_watermark_remove_task_status_model.reset()
            return
        if os.path.isdir(file_path):
            tmp_file_path = os.path.join(file_path, os.listdir(file_path)[0])
            self.status_info_widget.show_batch_pipeline_widget()
        else:
            tmp_file_path = file_path
            self.status_info_widget.show_pipeline_widget()
        file_type = get_file_type(tmp_file_path)
        ext = tmp_file_path.lower().split(".")[-1]
        if file_type == "image":
            self.stack.setCurrentIndex(1)
            self.media_type = "image"
        else:
            self.placeholder_widget.setText(f"不支持的文件类型: {ext}")
            self.stack.setCurrentIndex(0)
            blind_watermark_remove_task_status_model.reset()

    def update_preview(self, input_path, output_path):
        self.files_preview_info[input_path] = output_path
        self.image_navigation_widget.load_images([input_path], self.media_type)

    def _on_preview_file(self, path):
        out = self.files_preview_info.get(path)
        if self.media_type == "image" and out:
            self.image_viewer.set_images(img1=path, img2=out)


class HeaderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        header = GradientHeader(parent=self, start=QColor(139, 92, 246), stop=QColor(109, 40, 217))
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)
        header_layout.setSpacing(10)

        title_label = QLabel(self.tr("🔐 暗水印去除"))
        setFont(title_label, fontSize=24, weight=QFont.DemiBold)
        title_label.setStyleSheet("QLabel { color: white; }")
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        self.process_btn = PushButton(text=self.tr("▶️ 开始处理"))
        self.process_btn.setStyleSheet("""
            PushButton {
                background-color: rgba(255, 255, 255, 0.55);
                color: #4c1d95;
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

        self.process_btn.clicked.connect(self._on_process)

    def _on_process(self):
        params = blind_watermark_remove_params.to_dict()
        error_msg = self._params_check(params)
        if error_msg:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        # Worker not yet implemented - show placeholder tip
        TeachingTip.create(
            target=self.process_btn,
            icon=InfoBarIcon.INFORMATION,
            title=self.tr("提示"),
            content=self.tr("暗水印去除功能即将上线，敬请期待"),
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2000,
            parent=self
        )

    def _params_check(self, params):
        if not params:
            return self.tr("请设置暗水印去除参数")
        if not cfg.get(cfg.localBlindWatermarkEnabled):
            return self.tr("请在设置页面打开 '盲水印AI能力' 开关")
        if "input_path" not in params or not params["input_path"]:
            return self.tr("请选择要处理的文件或目录")
        if isinstance(params["input_path"], str) and " " in params["input_path"]:
            return self.tr("输入路径不能包含空格")
        if isinstance(params["input_path"], str) and not params["input_path"].isascii():
            return self.tr("输入路径: 不支持非英文路径")
        if "crop_size" not in params:
            return self.tr("请设置图像裁剪大小")
        if "model_name" not in params or not params["model_name"]:
            return self.tr("请选择去除模型")
        if "output_path" not in params or not params["output_path"]:
            return self.tr("请设置文件保存位置")
        if isinstance(params["output_path"], str) and not params["output_path"].isascii():
            return self.tr("输出路径: 不支持非英文路径")
        return ""


class ImageEdit(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageEdit")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = HeaderWidget(self)
        main_layout.addWidget(header, 0, Qt.AlignTop)

        view_layout = QHBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        control_panel_widget = ControlPanelWidget(self)
        view_layout.addWidget(control_panel_widget, 3)

        right_content = PreviewWidget(self)
        view_layout.addWidget(right_content, 7)

        main_layout.addLayout(view_layout)
