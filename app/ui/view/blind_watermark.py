import os
import sys
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (
    QVBoxLayout, QWidget, QLabel, QHBoxLayout, QStackedWidget, QStackedLayout, QLineEdit, QFileDialog,
    QButtonGroup, QPushButton
)
from PySide6.QtGui import QFont, QColor, QAction

from app.ui.library.qfluentwidgets import (
    setFont, HeaderCardWidget, SegmentedWidget, ScrollArea, PushButton, CaptionLabel,
    LineEdit, FluentIcon, TeachingTip, InfoBarIcon, TeachingTipTailPosition,
    MessageBox
)

from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.file_selector_widget import FileSelectorWidget
from app.ui.widgets.directory_selector_widget import DirectorySelectorWidget
from app.ui.widgets.custom_card_group_widget import StyleCard
from app.ui.widgets.image_preview_widget import SyncImageViewer, ImageNavigationWidget
from app.ui.widgets.status_bar_widget import StatusInfoWidget
from app.ui.widgets.task_info_messagebox_widget import TaskInfoMessageBox
from app.ui.widgets.toggle_switch_widget import ToggleSwitch
from app.ui.widgets.watermark_interactive_widget import AreaSelectorDialog

from app.ui.common.event_bus import global_event_bus
from app.ui.common.task_params import bind_widget_to_param, TaskParams
from app.ui.common.task_status import TaskStatusModel
from app.ui.common.utils import get_file_type, global_backend_info_cache
from app.ui.common.config import cfg

from app.controllers.task_manager import global_task_manager
from app.workers.blind_watermark_remove_work import BlindWatermarkRemoveWork

from app.license.globals import feature_gate


blind_watermark_remove_params = TaskParams()
blind_watermark_remove_task_status_model = TaskStatusModel()
# 当前批次已提交的任务 future，用于支持一次性取消
blind_watermark_remove_active_futures = []


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
        self.pivot.currentItemChanged.connect(lambda k: self.stackedWidget.setCurrentWidget(self.findChild(QWidget, k)))

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(routeKey=objectName, text=text)


class ProcessRegionCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🎯 原图保留区域（可选）"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        self.select_button = PushButton(text=self.tr("🖼 框选原图保留区域"))
        self.select_button.setEnabled(False)
        self.select_button.setStyleSheet("""
            PushButton { padding: 6px 12px; border-radius: 6px; font-size: 12px; }
            PushButton {
                background: #F1F5F9;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                padding: 6px;
                color: #0F172A;
            }
            PushButton:hover {
                background: #E2E8F0;
                border-color: #CBD5E1;
            }
            PushButton:pressed {
                background: #CBD5E1;
                padding-top: 7px;
                padding-bottom: 5px;
            }
        """)
        self.select_button.clicked.connect(self._on_select_region)
        main_layout.addWidget(self.select_button)

        self.region_info_label = CaptionLabel(self.tr("未选择区域，将处理整张图片"))
        self.region_info_label.setWordWrap(True)
        self.region_info_label.setStyleSheet("color: #999999;")
        main_layout.addWidget(self.region_info_label)

        hint_label = CaptionLabel(self.tr("仅支持单张图片；框选区域保留原图，其他区域使用处理结果"))
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #999999;")
        main_layout.addWidget(hint_label)

        self._file_path = ""
        global_event_bus.blindWatermarkRemove_InputFileUpdate.connect(self._on_file_selected)

    def _on_file_selected(self, file_path):
        self._file_path = file_path or ""
        blind_watermark_remove_params.set_param("reserve_region", None)
        self.region_info_label.setText(self.tr("未选择区域，将处理整张图片"))
        can_select = bool(
            self._file_path
            and os.path.isfile(self._file_path)
            and get_file_type(self._file_path) == "image"
        )
        self.select_button.setEnabled(can_select)
        if self._file_path and os.path.isdir(self._file_path):
            self.region_info_label.setText(self.tr("目录任务不使用区域框选，将处理每张图片"))
        elif self._file_path and not can_select:
            self.region_info_label.setText(self.tr("当前文件不是可框选的图片"))

    def _on_select_region(self):
        if not self._file_path:
            return
        dialog = AreaSelectorDialog(
            file_path=self._file_path,
            single_area_only=True,
            parent=self.window(),
        )
        dialog.exec()
        boxes = dialog.get_results()
        region = boxes[0] if boxes else None
        blind_watermark_remove_params.set_param("reserve_region", region)
        if region:
            self.region_info_label.setText(self.tr("已选择原图保留区域: X={0}, Y={1}, 宽={2}, 高={3}").format(
                    region["x"], region["y"], region["w"], region["h"]
                )
            )
        else:
            self.region_info_label.setText(self.tr("未选择区域，将处理整张图片"))


class ColorFixCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🎨 颜色修复"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        toggle_layout = QHBoxLayout()
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(10)

        label = QLabel(self.tr("启用颜色修复"))
        label.setStyleSheet("color: #666666;")
        toggle_layout.addWidget(label)
        toggle_layout.addStretch()
        self.toggle = ToggleSwitch(self)
        self.toggle.setActive(True, animated=False)
        toggle_layout.addWidget(self.toggle)

        bind_widget_to_param(self.toggle, "toggled", blind_watermark_remove_params, "use_color_fix", transform=None)
        self.toggle.toggled.emit(True)

        main_layout.addLayout(toggle_layout)

        hint_label = CaptionLabel(self.tr("确保输出图片和输入图片的色彩空间一致"))
        hint_label.setStyleSheet("color: #999999;")
        main_layout.addWidget(hint_label)


class HighQualityOutputCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("✨ 输出质量"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        toggle_layout = QHBoxLayout()
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(10)

        label = QLabel(self.tr("启用高质量输出"))
        label.setStyleSheet("color: #666666;")
        toggle_layout.addWidget(label)
        toggle_layout.addStretch()
        self.toggle = ToggleSwitch(self)
        toggle_layout.addWidget(self.toggle)

        bind_widget_to_param(
            self.toggle,
            "toggled",
            blind_watermark_remove_params,
            "high_quality_output",
            transform=None,
        )
        self.toggle.toggled.emit(False)
        main_layout.addLayout(toggle_layout)

        hint_label = CaptionLabel(self.tr("提高输出质量，增加暗印去除概率，但耗时可能翻倍。"))
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #999999;")
        main_layout.addWidget(hint_label)


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

        self.all_cards: list[StyleCard] = []

        # 图片模型列表
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 6, 0, 6)
        image_layout.setSpacing(0)

        reverse_edit_card = StyleCard("#8b5cf6", self.tr("智能消除"), self.tr("智能去除暗水印，无需原始水印信息"))
        reverse_edit_card.set_name("reverse_edit")
        image_layout.addWidget(reverse_edit_card)
        image_layout.addStretch()
        self.stacked_widget.addWidget(image_container)

        self.image_cards = [reverse_edit_card]
        self.all_cards.extend(self.image_cards)

        # 视频模型列表（占位）
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 6, 0, 6)
        video_layout.setSpacing(0)

        video_placeholder = QLabel(self.tr("视频暗水印去除模型开发中，敬请期待..."))
        video_placeholder.setStyleSheet("color: #999999; padding: 20px;")
        video_placeholder.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(video_placeholder)
        video_layout.addStretch()
        self.stacked_widget.addWidget(video_container)

        self.video_cards = []

        for card in self.all_cards:
            card.mousePressEvent = lambda event, c=card: self._on_card_clicked(c)

        self.tab_image.toggled.connect(lambda checked: self._on_tab_changed(0, checked))
        self.tab_video.toggled.connect(lambda checked: self._on_tab_changed(1, checked))

        bind_widget_to_param(self, "model_name", blind_watermark_remove_params, "model_name", transform=None)
        self.select_first_interactive()
        global_event_bus.License_update.connect(self.select_first_interactive)
        global_event_bus.blindWatermarkRemove_TaskFinishedByModel.connect(self._on_task_finished_by_model)
        main_layout.addStretch()

    def _on_tab_changed(self, index, checked):
        if not checked:
            return
        self.stacked_widget.setCurrentIndex(index)
        target_pool = self.image_cards if index == 0 else self.video_cards
        if target_pool:
            self._on_card_clicked(target_pool[0])
        content_height = self._get_current_page_height(index)
        extra_height = 114
        self._animate_height_change(content_height + extra_height)

    def _on_card_clicked(self, clicked_card):
        if not clicked_card.is_interactive():
            return
        current_index = self.stacked_widget.currentIndex()
        active_pool = self.image_cards if current_index == 0 else self.video_cards
        for c in active_pool:
            c.set_selected(False)
        clicked_card.set_selected(True)
        self.model_name.emit(clicked_card.get_name())

    def select_first_interactive(self):
        current_index = self.stacked_widget.currentIndex()
        active_pool = self.image_cards if current_index == 0 else self.video_cards
        if any(c.is_selected for c in active_pool):
            return
        for card in active_pool:
            if card.is_interactive():
                card.set_selected(True)
                self.model_name.emit(card.get_name())
                return

    def _on_task_finished_by_model(self, model_name):
        current_index = self.stacked_widget.currentIndex()
        active_pool = self.image_cards if current_index == 0 else self.video_cards
        for one_card in active_pool:
            if one_card.get_name() != model_name:
                continue
            one_card.UpdateLicenseInfo.emit()
            if not one_card.is_interactive():
                self.model_name.emit("")
                self.select_first_interactive()
            break

    def _animate_height_change(self, target_height: int):
        start_height = self.height()
        self.height_anim = QPropertyAnimation(self, b"maximumHeight")
        self.height_anim.setDuration(250)
        self.height_anim.setStartValue(start_height)
        self.height_anim.setEndValue(target_height)
        self.height_anim.setEasingCurve(QEasingCurve.OutCubic)
        self.height_anim.start()

    def _get_current_page_height(self, index: int):
        widget = self.stacked_widget.widget(index)
        widget.layout().activate()
        return widget.layout().sizeHint().height()


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
        main_layout.addWidget(ProcessRegionCard(self))
        main_layout.addWidget(ColorFixCard(self))
        main_layout.addWidget(HighQualityOutputCard(self))
        main_layout.addWidget(ModelSelectCard(self))
        main_layout.addWidget(OutputSettingsCard(self))
        main_layout.addStretch(1)

        self.setWidget(view)
        self.setViewportMargins(0, 0, 0, 0)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BlindWatermarkRemovePreview")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        self.stack = QStackedLayout()
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.placeholder_widget = QLabel(self.tr("请选择图片文件进行暗水印去除预览"), parent=self)
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
        self.status_info_widget.model.set_pipeline_steps(names=[self.tr('准备任务'), self.tr('暗水印去除'), self.tr('导出结果')])
        self.status_info_widget.cancel_requested.connect(self._cancel_tasks)
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
            self.placeholder_widget.setText(self.tr(f"不支持的文件类型: {ext}"))
            self.stack.setCurrentIndex(0)
            blind_watermark_remove_task_status_model.reset()

    def update_preview(self, input_path, result_tuple):
        # result_tuple 为 (output_path, metrics)
        output_path = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple
        self.files_preview_info[input_path] = output_path
        self.image_navigation_widget.load_images([input_path], self.media_type)

    def _on_preview_file(self, path):
        out = self.files_preview_info.get(path)
        if self.media_type == "image" and out:
            self.image_viewer.set_images(img1=path, img2=out)

    def _cancel_tasks(self):
        cancelled = 0
        for future in list(blind_watermark_remove_active_futures):
            if not future.done and global_task_manager.cancel(future.job_id):
                cancelled += 1
        TeachingTip.create(
            target=self.status_info_widget,
            icon=InfoBarIcon.WARNING,
            title=self.tr("通知"),
            content=self.tr("正在取消 {0} 个任务，正在运行的任务将被强制结束").format(cancelled),
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2500,
            parent=self
        )


class HeaderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        header = GradientHeader(parent=self, start=QColor(168, 237, 234), stop=QColor(254, 214, 227))
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

        self.is_batch_task = False
        self.process_btn.clicked.connect(self._on_process)

    def _on_process(self):
        init_params = blind_watermark_remove_params.to_dict()
        error_msg, task_params = self._params_check(params=init_params)
        if error_msg:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        w = TaskInfoMessageBox(task_params, "blind-watermark-remove", self.window())
        if not w.exec():
            return
        allowed_use, error_msg = feature_gate.can_use(
            feature_name=feature_gate.get_feature_name(blind_watermark_remove_params.to_dict()["model_name"]),
            return_errmsg=True
        )
        if not allowed_use:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        task_params["_feature_name_"] = feature_gate.get_feature_name( blind_watermark_remove_params.to_dict()["model_name"])
        total_tasks = []
        input_path = task_params["input_path"]
        global_event_bus.blindWatermarkRemove_ImageNavigationInit.emit()
        if os.path.isdir(input_path):
            self.is_batch_task = True
            for one_file in os.listdir(input_path):
                task_params["input_path"] = os.path.join(input_path, one_file)
                task_instance = BlindWatermarkRemoveWork(**task_params)
                func, args, kwargs = task_instance.to_worker()
                total_tasks.append((func, args, kwargs))
        else:
            self.is_batch_task = False
            task_instance = BlindWatermarkRemoveWork(**task_params)
            func, args, kwargs = task_instance.to_worker()
            total_tasks.append((func, args, kwargs))

        backend_type, gpu_name = global_backend_info_cache.get(key="backend_info")
        blind_watermark_remove_task_status_model.start_batch(total=len(total_tasks), backend_type=backend_type, gpu_name=gpu_name)
        if not self.is_batch_task:
            blind_watermark_remove_task_status_model.start_step(name=self.tr("准备任务"))
        blind_watermark_remove_active_futures.clear()
        for func, args, kwargs in total_tasks:
            input_path = kwargs["input_path"]
            future = global_task_manager.submit(func, *args, **kwargs)
            blind_watermark_remove_active_futures.append(future)
            future.finished.connect(
                lambda result, path=input_path: self._task_finished(path, result)
            )
            future.failed.connect(
                lambda e, path=input_path: blind_watermark_remove_task_status_model.report_failure(path, e)
            )
            future.cancelled.connect(
                lambda path=input_path: blind_watermark_remove_task_status_model.report_failure(path, "任务被取消")
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
        if value == "BlindWatermarkRemoveStart":
            if not self.is_batch_task:
                blind_watermark_remove_task_status_model.finish_step(name=self.tr("准备任务"))
                blind_watermark_remove_task_status_model.start_step(name=self.tr("暗水印去除"))
        elif value == "BlindWatermarkRemoveCompleted":
            if not self.is_batch_task:
                blind_watermark_remove_task_status_model.finish_step(name=self.tr("暗水印去除"))
                blind_watermark_remove_task_status_model.start_step(name=self.tr("导出结果"))

    def _task_finished(self, input_path, result):
        if not feature_gate.is_pro:
            try:
                feature_gate.use_feature(feature_name=feature_gate.get_feature_name(blind_watermark_remove_params.to_dict()["model_name"]))
            except Exception:
                pass
            global_event_bus.blindWatermarkRemove_TaskFinishedByModel.emit(blind_watermark_remove_params.to_dict()["model_name"])
        blind_watermark_remove_task_status_model.report_success()
        # result 为 (output_path, metrics) 元组
        output_tuple = result if isinstance(result, tuple) else (result, {})
        global_event_bus.blindWatermarkRemove_TaskFinished.emit(input_path, output_tuple)
        if not self.is_batch_task:
            blind_watermark_remove_task_status_model.finish_step(name=self.tr("导出结果"))

    def _params_check(self, params):
        error_msg = ""
        task_params = {}
        if not params:
            error_msg = self.tr("请设置暗水印去除参数")
            return error_msg, task_params
        if not cfg.get(cfg.localBlindWatermarkEnabled):
            error_msg = self.tr("请在设置页面打开 '盲水印AI能力' 开关")
            return error_msg, task_params
        if "input_path" not in params or not params["input_path"] or " " in params["input_path"]:
            error_msg = self.tr("请选择要处理的文件或目录且路径不能有空格")
            return error_msg, task_params
        if isinstance(params["input_path"], str) and not params["input_path"].isascii():
            return self.tr("输入路径: 不支持非英文路径"), task_params
        task_params["input_path"] = params["input_path"]
        if "model_name" not in params or not params["model_name"]:
            error_msg = self.tr("请选择去除模型")
            return error_msg, task_params
        task_params["model_name"] = params["model_name"]
        task_params["use_color_fix"] = params.get("use_color_fix", True)
        task_params["high_quality_output"] = params.get("high_quality_output", False)
        task_params["reserve_region"] = params.get("reserve_region")
        if "output_path" not in params or not params["output_path"]:
            error_msg = self.tr("请设置文件保存位置")
            return error_msg, task_params
        if isinstance(params["output_path"], str) and not params["output_path"].isascii():
            return self.tr("输出路径: 不支持非英文路径"), task_params
        task_params["output_path"] = params["output_path"]
        return error_msg, task_params


class BlindWatermarkRemove(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BlindWatermarkRemove")

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
