import os
import sys
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QWidget, QHBoxLayout, QLabel, QStackedWidget, QStackedLayout,
    QFrame, QButtonGroup, QPushButton, QLineEdit, QFileDialog, QTextEdit
)
from PySide6.QtGui import QFont, QColor, QAction, QFontMetrics

from app.ui.library.qfluentwidgets import (
    setFont, HeaderCardWidget, ScrollArea, PushButton, CaptionLabel, LineEdit, ComboBox,
    SpinBox, DoubleSpinBox, SegmentedWidget, FlowLayout, FluentIcon, TeachingTip, InfoBarIcon,
    TeachingTipTailPosition, MessageBox
)

from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.file_selector_widget import FileSelectorWidget
from app.ui.widgets.directory_selector_widget import DirectorySelectorWidget
from app.ui.widgets.image_preview_widget import SyncImageViewer, ImageNavigationWidget
from app.ui.widgets.status_bar_widget import StatusInfoWidget
from app.ui.widgets.task_info_messagebox_widget import TaskInfoMessageBox
from app.ui.widgets.custom_card_group_widget import CustomCardGroupWidget, StyleCard, CardSeparator
from app.ui.widgets.toggle_switch_widget import ToggleSwitch

from app.ui.common.event_bus import global_event_bus
from app.controllers.task_manager import global_task_manager
from app.ui.common.task_params import bind_widget_to_param, TaskParams
from app.ui.common.task_status import TaskStatusModel
from app.ui.common.utils import get_file_type, global_backend_info_cache
from app.ui.common.config import cfg

from app.workers.image_restoration_work import ImageRestorationWork

from app.license.globals import feature_gate


image_restoration_params = TaskParams()
image_restoration_task_status_model = TaskStatusModel()
image_restoration_active_futures = []
IMAGE_TASK_TYPES = (
    ("restoration", "✨", "综合修复", "自动判断退化类型，整体重建画质", "#667eea"),
    ("dehaze", "🌫️", "去雾", "去除雾霾，恢复通透的对比度", "#38bdf8"),
    ("derain", "🌧️", "去雨", "去除雨丝、雨点，还原背景细节", "#0ea5e9"),
    ("denoise", "🧹", "去噪", "抑制噪点与颗粒，保持纹理自然", "#22c55e"),
    ("deblur", "🔎", "去模糊", "修正抖动与失焦，锐化边缘", "#f59e0b"),
    ("super_resolution", "🔬", "超分放大", "放大分辨率并补充细节", "#a855f7"),
    ("lowlight", "💡", "低光增强", "提亮欠曝画面，恢复暗部细节", "#fbbf24"),
    ("compression", "🧩", "去压缩伪影", "消除块状噪声与振铃", "#14b8a6"),
    ("old_photo", "🖼️", "老照片修复", "修补划痕、褪色与污损", "#f472b6"),
    ("watermark_remove", "💧", "去水印", "去除 Logo、文字水印并补全背景", "#6366f1"),
    ("subtitle_remove", "🅰️", "去字幕", "擦除硬字幕并还原被覆盖内容", "#fb7185"),
)

TASK_TYPE_NAME_MAP = {key: name for key, _icon, name, _desc, _color in IMAGE_TASK_TYPES}


class TaskTypeCardItem(QFrame):
    clicked = Signal(str)
    NORMAL_STYLE = """
        TaskTypeCardItem {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
        }
        TaskTypeCardItem:hover {
            background-color: #f8fafc;
            border-color: #c7d2fe;
        }
    """
    SELECTED_STYLE = """
        TaskTypeCardItem {
            background-color: #eef2ff;
            border: 2px solid #667eea;
            border-radius: 10px;
        }
    """

    def __init__(self, key: str, icon: str, name: str, description: str, color: str, parent=None):
        super().__init__(parent)
        self.key = key
        self.is_selected = False
        self.setFixedSize(105, 45)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{name}：{description}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        icon_label = QLabel(icon, self)
        icon_label.setFixedSize(20, 20)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                border-radius: 8px;
                font-size: 14px;
            }}
        """)
        layout.addWidget(icon_label, 0, Qt.AlignVCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(0)

        title_label = QLabel(name, self)
        setFont(title_label, 12)
        title_label.setStyleSheet("color: #1f2937; background: transparent;")
        text_layout.addWidget(title_label)

        layout.addLayout(text_layout)

        self.setStyleSheet(self.NORMAL_STYLE)

    def set_selected(self, selected: bool):
        self.is_selected = bool(selected)
        self.setStyleSheet(self.SELECTED_STYLE if self.is_selected else self.NORMAL_STYLE)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.key)
            event.accept()
            return
        super().mousePressEvent(event)


class TaskTypeCard(HeaderCardWidget):
    task_type = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🧰 修复任务"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        flow_layout = FlowLayout(needAni=True)
        flow_layout.setContentsMargins(0, 0, 0, 0)
        flow_layout.setHorizontalSpacing(8)
        flow_layout.setVerticalSpacing(8)

        self.cards: list[TaskTypeCardItem] = []
        for key, icon, name, description, color in IMAGE_TASK_TYPES:
            card = TaskTypeCardItem(key, icon, name, description, color, self)
            card.clicked.connect(self._on_card_clicked)
            flow_layout.addWidget(card)
            self.cards.append(card)
        main_layout.addLayout(flow_layout)

        bind_widget_to_param(self, "task_type", image_restoration_params, "task_type", transform=None)
        self._on_card_clicked(self.cards[0].key)

    def _on_card_clicked(self, key: str):
        for card in self.cards:
            card.set_selected(card.key == key)
        self.task_type.emit(key)


class ModelSelectCard(HeaderCardWidget):
    model_name = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🤖 修复模型"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        tab_widget = QWidget(self)
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
            QPushButton:disabled {
                color: #bdc3c7;
            }
        """)

        self.tab_group = QButtonGroup(self)
        self.tab_group.setExclusive(True)
        self.tab_image = QPushButton(self.tr("图片模型"), tab_widget)
        setFont(self.tab_image, fontSize=14, weight=QFont.DemiBold)
        self.tab_image.setCheckable(True)
        self.tab_image.setChecked(True)
        self.tab_video = QPushButton(self.tr("视频模型"), tab_widget)
        setFont(self.tab_video, fontSize=14, weight=QFont.DemiBold)
        self.tab_video.setCheckable(True)
        self.tab_video.setEnabled(False)
        self.tab_video.setToolTip(self.tr("视频修复模型接入中，敬请期待"))
        self.tab_group.addButton(self.tab_image, 0)
        self.tab_group.addButton(self.tab_video, 1)
        tab_layout.addWidget(self.tab_image)
        tab_layout.addWidget(self.tab_video)
        main_layout.addWidget(tab_widget)

        self.stacked_widget = QStackedWidget(self)
        main_layout.addWidget(self.stacked_widget)

        # 图片模型
        image_container = QWidget(self)
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 6, 0, 6)
        image_layout.setSpacing(0)

        restoration_card = StyleCard("#f093fb", self.tr("智能修复"), self.tr("统一的图像修复模型，覆盖多种退化场景，速度较慢"))
        restoration_card.set_name("image_restoration")
        image_layout.addWidget(restoration_card)
        image_layout.addWidget(CardSeparator(self))

        super_resolution_card = StyleCard("#84fab0", self.tr("图像高清"), self.tr("提高图像分辨率，增强图像清晰度和细节，速度较慢"))
        super_resolution_card.set_name("image_sr")
        image_layout.addWidget(super_resolution_card)
        image_layout.addWidget(CardSeparator(self))

        image_layout.addStretch()
        self.stacked_widget.addWidget(image_container)
        self.image_cards: list[StyleCard] = [restoration_card, super_resolution_card]

        # 视频模型（占位，待算法接入）
        video_container = QWidget(self)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 6, 0, 6)
        video_layout.setSpacing(0)
        video_placeholder = QLabel(self.tr("视频修复模型接入中，敬请期待..."), video_container)
        video_placeholder.setAlignment(Qt.AlignCenter)
        video_placeholder.setStyleSheet("color: #999999; padding: 20px;")
        video_layout.addWidget(video_placeholder)
        video_layout.addStretch()
        self.stacked_widget.addWidget(video_container)
        self.video_cards: list[StyleCard] = []

        self.all_cards = list(self.image_cards) + list(self.video_cards)
        for card in self.all_cards:
            card.mousePressEvent = lambda event, c=card: self._on_card_clicked(c)

        self.tab_image.toggled.connect(lambda checked: self._on_tab_changed(0, checked))
        self.tab_video.toggled.connect(lambda checked: self._on_tab_changed(1, checked))

        bind_widget_to_param(self, "model_name", image_restoration_params, "model_name", transform=None)
        self._update_card_interactive()
        self.select_first_interactive()
        global_event_bus.License_update.connect(lambda: (self._update_card_interactive(), self.select_first_interactive()))
        global_event_bus.imageRestoration_TaskFinishedByModel.connect(self._on_task_finished_by_model)
        main_layout.addStretch()

        image_restoration_params.param_changed.connect(self._on_param_changed)

    def _update_card_interactive(self):
        if "task_type" not in image_restoration_params.to_dict():
            return
        is_super_resolution_task = True if image_restoration_params.to_dict()["task_type"] == "super_resolution" else False
        for card in self.all_cards:
            card.set_selected(False)
            if card.get_name() == "image_sr":
                card.set_interactive(is_super_resolution_task is True)
            else:
                card.set_interactive(is_super_resolution_task is False)
        
    def _on_param_changed(self, key, value):
        if key != "task_type":
            return
        self._update_card_interactive()
        self.select_first_interactive()

    @property
    def cards(self) -> list:
        return self.image_cards if self.stacked_widget.currentIndex() == 0 else self.video_cards

    def _on_tab_changed(self, index: int, checked: bool):
        if not checked:
            return
        self.stacked_widget.setCurrentIndex(index)
        if self.cards:
            self._on_card_clicked(self.cards[0])
        else:
            self.model_name.emit("")

    def _on_card_clicked(self, clicked_card):
        if not clicked_card.is_interactive():
            return
        for card in self.cards:
            card.set_selected(False)
        clicked_card.set_selected(True)
        self.model_name.emit(clicked_card.get_name())

    def select_first_interactive(self):
        if any(card.is_selected for card in self.cards):
            return
        for card in self.cards:
            if card.is_interactive():
                card.set_selected(True)
                self.model_name.emit(card.get_name())
                return
        self.model_name.emit("")

    def _on_task_finished_by_model(self, model_name):
        for card in self.all_cards:
            if card.get_name() != model_name:
                continue
            card.UpdateLicenseInfo.emit()
            if not card.is_interactive():
                self.model_name.emit("")
                self.select_first_interactive()
            break


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

        single_file_selector = FileSelectorWidget(self)
        single_file_selector.item_selected.connect(
            lambda file_path: global_event_bus.imageRestoration_InputFileUpdate.emit(file_path)
        )
        bind_widget_to_param(single_file_selector, "item_selected", image_restoration_params, "input_path", transform=None)

        batch_files_selector = DirectorySelectorWidget(self)
        batch_files_selector.item_selected.connect(
            lambda file_path: global_event_bus.imageRestoration_InputFileUpdate.emit(file_path)
        )
        bind_widget_to_param(batch_files_selector, "item_selected", image_restoration_params, "input_path", transform=None)

        self._add_sub_interface(single_file_selector, 'RestorationFileSelector', self.tr("文件"))
        self._add_sub_interface(batch_files_selector, 'RestorationDirectorySelector', self.tr("目录"))

        self.stackedWidget.setCurrentWidget(single_file_selector)
        self.pivot.setCurrentItem(single_file_selector.objectName())
        self.pivot.currentItemChanged.connect(
            lambda key: self.stackedWidget.setCurrentWidget(self.findChild(QWidget, key))
        )

    def _add_sub_interface(self, widget: QWidget, object_name: str, text: str):
        widget.setObjectName(object_name)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(routeKey=object_name, text=text)


class SettingsCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("⚙️ 修复参数"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        self.upscale_combox = ComboBox()
        setFont(self.upscale_combox, 14)
        self.upscale_combox.addItems(["4x", "3x", "2x"])
        self.upscale_card = CustomCardGroupWidget(
            title=self.tr("放大倍数"),
            content=self.tr("仅超分放大任务生效"),
            parent=self,
            text_layout_contents_margins=(12, 0, 0, 0),
            label_v_space=2
        )
        self.upscale_card.addWidget(self.upscale_combox, stretch=0)
        self.upscale_card.setSeparatorVisible(True)
        bind_widget_to_param(
            self.upscale_combox, "currentTextChanged", image_restoration_params, "upscale",
            transform=lambda text: int(str(text).rstrip("xX") or 1)
        )
        self.upscale_combox.currentTextChanged.emit("4x")
        self.upscale_card.hide()
        main_layout.addWidget(self.upscale_card)

        low_memory_btn = ToggleSwitch(on_color="#667eea")
        low_memory_btn.setActive(True)
        low_memory_card = CustomCardGroupWidget(
            title=self.tr("低显存模式"),
            content=self.tr("每次推理后释放模型显存，显存较小时建议开启"),
            parent=self,
            text_layout_contents_margins=(12, 0, 0, 0),
            label_v_space=2
        )
        low_memory_card.addWidget(low_memory_btn)
        low_memory_card.setContentWordWrap(True)
        low_memory_card.setSeparatorVisible(True)
        bind_widget_to_param(low_memory_btn, "toggled", image_restoration_params, "low_memory", transform=bool)
        low_memory_btn.toggled.emit(True)
        main_layout.addWidget(low_memory_card)

        prompt_layout = QVBoxLayout()
        prompt_layout.setContentsMargins(12, 10, 24, 6)
        prompt_layout.setSpacing(6)
        prompt_label = CaptionLabel(self.tr("自定义提示词（可选）"))
        setFont(prompt_label, 13)
        prompt_label.setStyleSheet("color: #888888;")
        prompt_layout.addWidget(prompt_label)

        self.prompt_edit = QTextEdit(self)
        self.prompt_edit.setMaximumHeight(72)
        self.prompt_edit.setPlaceholderText(self.tr("留空则使用所选任务的默认修复描述"))
        self.prompt_edit.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 6px;
                background-color: #fafafa;
                color: #1a1a1a;
            }
            QTextEdit:focus {
                border: 1px solid #667eea;
                background-color: #ffffff;
            }
        """)
        setFont(self.prompt_edit, fontSize=13)
        self.prompt_edit.textChanged.connect(lambda: image_restoration_params.set_param("prompt", self.prompt_edit.toPlainText()))
        prompt_layout.addWidget(self.prompt_edit)
        main_layout.addLayout(prompt_layout)

        image_restoration_params.param_changed.connect(self._on_param_changed)

    def _on_param_changed(self, key, value):
        if key != "task_type":
            return
        self.upscale_card.setVisible(value == "super_resolution")


class OutputSettingsCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("💾 输出设置"))
        self.setBorderRadius(8)
        self.viewLayout.setContentsMargins(10, 10, 10, 10)

        output_settings_layout = QVBoxLayout()
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
        bind_widget_to_param(self.save_location_line_edit, "textChanged", image_restoration_params, "output_path", transform=None)
        output_settings_layout.addWidget(self.save_location_line_edit)

        self.viewLayout.addLayout(output_settings_layout)

    def save_location_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择文件夹", "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog
            if sys.platform == "darwin" else QFileDialog.Option(0)
        )
        if directory:
            self.save_location_line_edit.setText(directory)


class ControlPanelWidget(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        view = QWidget(self)
        view.setObjectName('imageRestorationControlPanel')
        main_layout = QVBoxLayout(view)
        main_layout.setContentsMargins(0, 0, 12, 0)
        main_layout.setSpacing(10)
        main_layout.setAlignment(Qt.AlignTop)

        main_layout.addWidget(FileSelectorCard(self))
        main_layout.addWidget(TaskTypeCard(self))
        main_layout.addWidget(ModelSelectCard(self))
        main_layout.addWidget(SettingsCard(self))
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
        self.setObjectName("ImageRestorationPreview")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        self.stack = QStackedLayout()
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.placeholder_widget = QLabel(self.tr("请选择图片文件或目录进行图像修复"), parent=self)
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

        self.image_navigation_widget = ImageNavigationWidget(parent=self, task_type="image_restoration")
        bottom_layout.addWidget(self.image_navigation_widget, 3)

        self.status_info_widget = StatusInfoWidget(image_restoration_task_status_model, self)
        self.status_info_widget.model.set_pipeline_steps(
            names=[self.tr('准备任务'), self.tr('图像修复'), self.tr('导出结果')]
        )
        self.status_info_widget.cancel_requested.connect(self._cancel_tasks)
        bottom_layout.addWidget(self.status_info_widget, 2)

        main_layout.addLayout(bottom_layout)

        self.files_preview_info = {}
        self.media_type = "image"

        global_event_bus.imageRestoration_InputFileUpdate.connect(self.update_init_preview)
        global_event_bus.imageRestoration_TaskFinished.connect(self.update_preview)
        global_event_bus.imageRestoration_PreviewFile.connect(self._on_preview_file)
        global_event_bus.imageRestoration_ImageNavigationInit.connect(self._on_navigation_init)

    def _on_navigation_init(self):
        self.image_navigation_widget.clear_images()
        if self.stack.currentIndex() == 0:
            self.stack.setCurrentIndex(1)
            self.image_viewer.init_scene()

    def update_init_preview(self, file_path):
        self.image_navigation_widget.clear_images()
        self.image_viewer.init_scene()
        self.files_preview_info = {}
        self.media_type = "image"
        if not file_path:
            self.stack.setCurrentIndex(0)
            image_restoration_task_status_model.reset()
            return
        if os.path.isdir(file_path):
            files = os.listdir(file_path)
            if not files:
                self.placeholder_widget.setText(self.tr("所选目录为空，请重新选择"))
                self.stack.setCurrentIndex(0)
                image_restoration_task_status_model.reset()
                return
            tmp_file_path = os.path.join(file_path, files[0])
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
            image_restoration_task_status_model.reset()

    def update_preview(self, input_path, output_path):
        self.files_preview_info[input_path] = output_path
        if self.stack.currentIndex() == 0:
            self.stack.setCurrentIndex(1)
        self.image_navigation_widget.load_images([input_path], self.media_type)

    def _on_preview_file(self, path):
        out = self.files_preview_info.get(path)
        if self.media_type == "image" and out:
            self.image_viewer.set_images(img1=path, img2=out)

    def _cancel_tasks(self):
        cancelled = 0
        for future in list(image_restoration_active_futures):
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
        header = GradientHeader(parent=self, start=QColor(99, 102, 241), stop=QColor(168, 85, 247))
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)
        header_layout.setSpacing(10)

        title_label = QLabel(self.tr("🩹 图像修复"))
        setFont(title_label, fontSize=24, weight=QFont.DemiBold)
        title_label.setStyleSheet("QLabel { color: white; }")
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        self.process_btn = PushButton(text=self.tr("▶️ 开始处理"))
        self.process_btn.setStyleSheet("""
            PushButton {
                background-color: rgba(255, 255, 255, 0.55);
                color: #312e81;
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
        init_params = image_restoration_params.to_dict()
        error_msg, task_params = self._params_check(params=init_params)
        if error_msg:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return

        w = TaskInfoMessageBox(task_params, "image-restoration", self.window())
        if not w.exec():
            return

        allowed_use, error_msg, feature_name = self._license_check(task_params["model_name"])
        if not allowed_use:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        task_params["_feature_name_"] = feature_name

        total_tasks = []
        input_path = task_params["input_path"]
        global_event_bus.imageRestoration_ImageNavigationInit.emit()
        if os.path.isdir(input_path):
            self.is_batch_task = True
            for one_file in sorted(os.listdir(input_path)):
                one_file_path = os.path.join(input_path, one_file)
                if not os.path.isfile(one_file_path) or get_file_type(one_file_path) != "image":
                    continue
                task_params["input_path"] = one_file_path
                func, args, kwargs = ImageRestorationWork(**task_params).to_worker()
                total_tasks.append((func, args, kwargs))
        else:
            self.is_batch_task = False
            func, args, kwargs = ImageRestorationWork(**task_params).to_worker()
            total_tasks.append((func, args, kwargs))

        if not total_tasks:
            MessageBox(
                title=self.tr("提醒"), content=self.tr("所选目录中没有可处理的图片"), parent=self.window()
            ).exec()
            return

        backend_type, gpu_name = global_backend_info_cache.get(key="backend_info")
        image_restoration_task_status_model.start_batch(
            total=len(total_tasks), backend_type=backend_type, gpu_name=gpu_name
        )
        if not self.is_batch_task:
            image_restoration_task_status_model.start_step(name=self.tr("准备任务"))

        image_restoration_active_futures.clear()
        for func, args, kwargs in total_tasks:
            one_input_path = kwargs["input_path"]
            future = global_task_manager.submit(func, *args, **kwargs)
            image_restoration_active_futures.append(future)

            future.finished.connect(
                lambda result, path=one_input_path: self._task_finished(path, result)
            )
            future.failed.connect(
                lambda e, path=one_input_path: image_restoration_task_status_model.report_failure(path, e)
            )
            future.cancelled.connect(
                lambda path=one_input_path: image_restoration_task_status_model.report_failure(path, "任务被取消")
            )
            future.progress.connect(
                lambda value, msg, path=one_input_path: self._task_progress(path, value, msg)
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

    def _license_check(self, model_name):
        try:
            if feature_gate.is_pro:
                return True, "", ""
            feature_name = feature_gate.get_feature_name(model_name)
            allowed_use, error_msg = feature_gate.can_use(feature_name=feature_name, return_errmsg=True)
            return allowed_use, error_msg, feature_name
        except Exception as e:
            return False, self.tr("功能「图像修复」授权校验失败：{0}").format(e), ""

    def _task_progress(self, input_path, value, msg):
        if value == "RestorationStart":
            if not self.is_batch_task:
                image_restoration_task_status_model.finish_step(name=self.tr("准备任务"))
                image_restoration_task_status_model.start_step(name=self.tr("图像修复"))
        elif value == "RestorationCompleted":
            if not self.is_batch_task:
                image_restoration_task_status_model.finish_step(name=self.tr("图像修复"))
                image_restoration_task_status_model.start_step(name=self.tr("导出结果"))

    def _task_finished(self, input_path, result):
        model_name = image_restoration_params.to_dict().get("model_name", "")
        try:
            if not feature_gate.is_pro:
                try:
                    feature_gate.use_feature(feature_name=feature_gate.get_feature_name(model_name))
                except Exception:
                    pass
                global_event_bus.imageRestoration_TaskFinishedByModel.emit(model_name)
        except Exception:
            pass
        image_restoration_task_status_model.report_success()
        output_path = result[0] if isinstance(result, tuple) else result
        global_event_bus.imageRestoration_TaskFinished.emit(input_path, output_path)
        if not self.is_batch_task:
            image_restoration_task_status_model.finish_step(name=self.tr("导出结果"))

    def _params_check(self, params):
        error_msg = ""
        task_params = {}
        if not params:
            error_msg = self.tr("请设置图像修复参数")
            return error_msg, task_params
        if not cfg.get(cfg.localImageEditEnabled):
            error_msg = self.tr("请在设置页面打开 '图像编辑AI能力' 开关")
            return error_msg, task_params
        if "input_path" not in params or not params["input_path"]:
            error_msg = self.tr("请选择要处理的文件或目录")
            return error_msg, task_params
        task_params["input_path"] = params["input_path"]
        if "task_type" not in params or not params["task_type"]:
            error_msg = self.tr("请选择修复任务类型")
            return error_msg, task_params
        task_params["task_type"] = params["task_type"]
        if "model_name" not in params or not params["model_name"]:
            error_msg = self.tr("请选择修复模型")
            return error_msg, task_params
        task_params["model_name"] = params["model_name"]
        task_params["low_memory"] = bool(params.get("low_memory", True))
        if params["task_type"] == "super_resolution":
            task_params["upscale"] = int(params.get("upscale", 2))
        prompt = (params.get("prompt") or "").strip()
        if prompt:
            task_params["prompt"] = prompt
        if "output_path" not in params or not params["output_path"]:
            error_msg = self.tr("请设置文件保存位置")
            return error_msg, task_params
        task_params["output_path"] = params["output_path"]
        return error_msg, task_params


class ImageRestoration(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageRestoration")

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
