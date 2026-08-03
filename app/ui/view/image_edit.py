import os
import sys
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (
    QVBoxLayout, QWidget, QLabel, QHBoxLayout, QStackedWidget, QStackedLayout,
    QLineEdit, QFileDialog, QButtonGroup, QPushButton, QTextEdit
)
from PySide6.QtGui import QFont, QColor, QAction, QPixmap

from app.ui.library.qfluentwidgets import (
    setFont, HeaderCardWidget, ScrollArea, PushButton, CaptionLabel,
    LineEdit, FluentIcon, TeachingTip, InfoBarIcon, TeachingTipTailPosition,
    MessageBox
)

from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.file_selector_widget import FileSelectorWidget
from app.ui.widgets.custom_card_group_widget import StyleCard, CardSeparator
from app.ui.widgets.image_preview_widget import SyncImageViewer, ImageNavigationWidget
from app.ui.widgets.status_bar_widget import StatusInfoWidget
from app.ui.widgets.task_info_messagebox_widget import TaskInfoMessageBox

from app.ui.common.event_bus import global_event_bus
from app.controllers.task_manager import global_task_manager
from app.ui.common.task_params import bind_widget_to_param, TaskParams
from app.ui.common.task_status import TaskStatusModel
from app.ui.common.utils import get_file_type, global_backend_info_cache
from app.ui.common.config import cfg
from app.workers.image_edit_work import ImageEditWork
from app.license.globals import feature_gate


image_edit_params = TaskParams()
image_edit_task_status_model = TaskStatusModel()
# 当前批次已提交的任务 future，用于支持一次性取消
image_edit_active_futures = []


class PromptInputCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("✍️ 提示词"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        self.prompt_edit = QTextEdit(self)
        self.prompt_edit.setPlaceholderText(self.tr("输入编辑提示词【建议英文】，描述期望编辑效果..."))
        self.prompt_edit.setMaximumHeight(100)
        self.prompt_edit.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 8px;
                background-color: #fafafa;
            }
            QTextEdit:focus {
                border: 1px solid #667eea;
                background-color: #ffffff;
            }
        """)
        setFont(self.prompt_edit, fontSize=13)
        self.prompt_edit.textChanged.connect(self._on_text_changed)
        main_layout.addWidget(self.prompt_edit)

        hint_label = CaptionLabel(self.tr("描述对图片的编辑操作\n如: Remove watermark, Repair background."))
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #999999;")
        main_layout.addWidget(hint_label)

    def _on_text_changed(self):
        image_edit_params.set_param("prompt", self.prompt_edit.toPlainText())


class ReferenceImageCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🖼️ 参考图"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        self.file_selector = FileSelectorWidget(self)
        self.file_selector.item_selected.connect(self._on_file_selected)
        bind_widget_to_param(self.file_selector, "item_selected", image_edit_params, "input_path", transform=None)
        main_layout.addWidget(self.file_selector)

        preview_layout = QHBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)

        self.preview_label = QLabel(self)
        self.preview_label.setFixedSize(120, 90)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 1px dashed #ccc;
                border-radius: 8px;
                background-color: #f9f9f9;
                color: #999;
                font-size: 12px;
            }
        """)
        self.preview_label.setText(self.tr("无预览"))
        preview_layout.addWidget(self.preview_label)

        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(6)

        self.area_select_btn = PushButton(text=self.tr("区域选择"))
        self.area_select_btn.setEnabled(False)
        self.area_select_btn.setStyleSheet("""
            PushButton { padding: 6px 12px; border-radius: 6px; font-size: 12px; }
        """)
        self.area_select_btn.clicked.connect(self._on_area_select)
        btn_layout.addWidget(self.area_select_btn)

        self.area_info_label = CaptionLabel(self.tr("未选择区域"))
        self.area_info_label.setStyleSheet("color: #999999;")
        btn_layout.addWidget(self.area_info_label)
        btn_layout.addStretch()
        preview_layout.addLayout(btn_layout)
        preview_layout.addStretch()
        main_layout.addLayout(preview_layout)

        hint_label = CaptionLabel(self.tr("选择参考图后可框选需要编辑的区域（Mask）"))
        hint_label.setStyleSheet("color: #999999;")
        main_layout.addWidget(hint_label)

        self._file_path = ""

    def _on_file_selected(self, file_path):
        self._file_path = file_path
        image_edit_params.set_param("mask_boxes", [])
        self.area_info_label.setText(self.tr("未选择区域"))

        if file_path and os.path.isfile(file_path):
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.preview_label.setPixmap(
                    pixmap.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self.area_select_btn.setEnabled(True)
            else:
                self.preview_label.setText(self.tr("无预览"))
                self.area_select_btn.setEnabled(False)
        else:
            self.preview_label.setText(self.tr("无预览"))
            self.area_select_btn.setEnabled(False)

        global_event_bus.imageEdit_InputFileUpdate.emit(file_path)

    def _on_area_select(self):
        if not self._file_path:
            return
        from app.ui.widgets.watermark_interactive_widget import AreaSelectorDialog
        dialog = AreaSelectorDialog(
            file_path=self._file_path, single_area_only=False, parent=self.window()
        )
        if dialog.exec():
            boxes = dialog.saved_boxes
            image_edit_params.set_param("mask_boxes", boxes)
            if boxes:
                self.area_info_label.setText(self.tr("已选择 {0} 个区域").format(len(boxes)))
            else:
                self.area_info_label.setText(self.tr("未选择区域"))


class ModelSelectCard(HeaderCardWidget):
    model_name = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🤖 编辑模型"))
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
        self.tab_video.setEnabled(False)
        self.tab_group.addButton(self.tab_image, 0)
        self.tab_group.addButton(self.tab_video, 1)
        tab_layout.addWidget(self.tab_image)
        tab_layout.addWidget(self.tab_video)
        main_layout.addWidget(tab_widget)

        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        self.all_cards: list[StyleCard] = []

        # 图片模型
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 6, 0, 6)
        image_layout.setSpacing(0)

        general_edit_card = StyleCard("#667eea", self.tr("智能重绘"), self.tr("语义理解强，细节丰富，复杂场景效果佳"))
        general_edit_card.set_name("general_edit")
        image_layout.addWidget(general_edit_card)
        image_layout.addStretch()
        self.stacked_widget.addWidget(image_container)

        self.image_cards = [general_edit_card]
        self.all_cards.extend(self.image_cards)

        # 视频模型（占位）
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 6, 0, 6)
        video_layout.setSpacing(0)

        video_placeholder = QLabel(self.tr("视频编辑模型开发中，敬请期待..."))
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

        bind_widget_to_param(self, "model_name", image_edit_params, "model_name", transform=None)
        self.select_first_interactive()
        global_event_bus.License_update.connect(self.select_first_interactive)
        global_event_bus.imageEdit_TaskFinishedByModel.connect(self._on_task_finished_by_model)
        main_layout.addStretch()

    def _on_tab_changed(self, index, checked):
        if not checked:
            return
        self.stacked_widget.setCurrentIndex(index)
        target_pool = self.image_cards if index == 0 else self.video_cards
        if target_pool:
            self._on_card_clicked(target_pool[0])
        content_height = self._get_current_page_height(index)
        self._animate_height_change(content_height + 114)

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
        bind_widget_to_param(
            self.save_location_line_edit, "textChanged", image_edit_params, "output_path", transform=None
        )
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
        view.setObjectName('imageEditControlPanel')
        main_layout = QVBoxLayout(view)
        main_layout.setContentsMargins(0, 0, 12, 0)
        main_layout.setSpacing(10)
        main_layout.setAlignment(Qt.AlignTop)

        main_layout.addWidget(PromptInputCard(self))
        main_layout.addWidget(ReferenceImageCard(self))
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
        self.setObjectName("ImageEditPreview")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        self.stack = QStackedLayout()
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.placeholder_widget = QLabel(self.tr("请选择参考图并输入提示词进行图像编辑"), parent=self)
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

        self.image_navigation_widget = ImageNavigationWidget(parent=self, task_type="image_edit")
        bottom_layout.addWidget(self.image_navigation_widget, 3)

        self.status_info_widget = StatusInfoWidget(image_edit_task_status_model, self)
        self.status_info_widget.model.set_pipeline_steps(
            names=[self.tr('准备任务'), self.tr('图像编辑'), self.tr('导出结果')]
        )
        self.status_info_widget.cancel_requested.connect(self._cancel_tasks)
        bottom_layout.addWidget(self.status_info_widget, 2)

        main_layout.addLayout(bottom_layout)

        self.files_preview_info = {}
        self.media_type = "image"

        global_event_bus.imageEdit_InputFileUpdate.connect(self.update_init_preview)
        global_event_bus.imageEdit_TaskFinished.connect(self.update_preview)
        global_event_bus.imageEdit_PreviewFile.connect(self._on_preview_file)
        global_event_bus.imageEdit_ImageNavigationInit.connect(
            lambda: self.image_navigation_widget.clear_images()
        )

    def update_init_preview(self, file_path):
        self.image_navigation_widget.clear_images()
        self.image_viewer.init_scene()
        self.files_preview_info = {}
        self.media_type = "image"
        if not file_path:
            self.stack.setCurrentIndex(0)
            image_edit_task_status_model.reset()
            return
        self.status_info_widget.show_pipeline_widget()
        file_type = get_file_type(file_path)
        ext = file_path.lower().split(".")[-1]
        if file_type == "image":
            self.stack.setCurrentIndex(1)
            self.media_type = "image"
        else:
            self.placeholder_widget.setText(self.tr(f"不支持的文件类型: {ext}"))
            self.stack.setCurrentIndex(0)
            image_edit_task_status_model.reset()

    def update_preview(self, input_path, output_path):
        self.files_preview_info[input_path] = output_path
        self.image_navigation_widget.load_images([input_path], self.media_type)

    def _on_preview_file(self, path):
        out = self.files_preview_info.get(path)
        if self.media_type == "image" and out:
            self.image_viewer.set_images(img1=path, img2=out)

    def _cancel_tasks(self):
        cancelled = 0
        for future in list(image_edit_active_futures):
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
        header = GradientHeader(parent=self, start=QColor(102, 126, 234), stop=QColor(118, 75, 162))
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)
        header_layout.setSpacing(10)

        title_label = QLabel(self.tr("🎨 图像编辑"))
        setFont(title_label, fontSize=24, weight=QFont.DemiBold)
        title_label.setStyleSheet("QLabel { color: white; }")
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        self.process_btn = PushButton(text=self.tr("▶️ 开始处理"))
        self.process_btn.setStyleSheet("""
            PushButton {
                background-color: rgba(255, 255, 255, 0.55);
                color: #3b0764;
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
        init_params = image_edit_params.to_dict()
        error_msg, task_params = self._params_check(params=init_params)
        if error_msg:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        w = TaskInfoMessageBox(task_params, "image-edit", self.window())
        if not w.exec():
            return
        allowed_use, error_msg = feature_gate.can_use(
            feature_name=feature_gate.get_feature_name(image_edit_params.to_dict()["model_name"]),
            return_errmsg=True
        )
        if not allowed_use:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        task_params["_feature_name_"] = feature_gate.get_feature_name(image_edit_params.to_dict()["model_name"])
        input_path = task_params["input_path"]
        output_dir = task_params["output_path"]
        basename = os.path.basename(input_path).rsplit(".", 1)
        output_file = os.path.join(
            output_dir,
            "{0}_edited.{1}".format(basename[0], basename[1] if len(basename) > 1 else "png")
        )
        task_params["output_path"] = output_file

        global_event_bus.imageEdit_ImageNavigationInit.emit()
        self.is_batch_task = False

        task_instance = ImageEditWork(**task_params)
        func, args, kwargs = task_instance.to_worker()

        backend_type, gpu_name = global_backend_info_cache.get(key="backend_info")
        image_edit_task_status_model.start_batch(total=1, backend_type=backend_type, gpu_name=gpu_name)
        image_edit_task_status_model.start_step(name=self.tr("准备任务"))

        image_edit_active_futures.clear()
        future = global_task_manager.submit(func, *args, **kwargs)
        image_edit_active_futures.append(future)

        future.finished.connect(
            lambda result, path=input_path: self._task_finished(path, result)
        )
        future.failed.connect(
            lambda e, path=input_path: image_edit_task_status_model.report_failure(path, e)
        )
        future.cancelled.connect(
            lambda path=input_path: image_edit_task_status_model.report_failure(path, "任务被取消")
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
        if value == "EditStart":
            image_edit_task_status_model.finish_step(name=self.tr("准备任务"))
            image_edit_task_status_model.start_step(name=self.tr("图像编辑"))
        elif value == "EditDone":
            image_edit_task_status_model.finish_step(name=self.tr("图像编辑"))
            image_edit_task_status_model.start_step(name=self.tr("导出结果"))

    def _task_finished(self, input_path, output_path):
        if not feature_gate.is_pro:
            try:
                feature_gate.use_feature(
                    feature_name=feature_gate.get_feature_name(
                        image_edit_params.to_dict()["model_name"]
                    )
                )
            except Exception:
                pass
            global_event_bus.imageEdit_TaskFinishedByModel.emit(
                image_edit_params.to_dict()["model_name"]
            )
        image_edit_task_status_model.report_success()
        global_event_bus.imageEdit_TaskFinished.emit(input_path, output_path)
        image_edit_task_status_model.finish_step(name=self.tr("导出结果"))

    def _params_check(self, params):
        error_msg = ""
        task_params = {}
        if not params:
            error_msg = self.tr("请设置图像编辑参数")
            return error_msg, task_params
        if not cfg.get(cfg.localImageEditEnabled):
            error_msg = self.tr("请在设置页面打开 '图像编辑AI能力' 开关")
            return error_msg, task_params
        if "prompt" not in params or not params.get("prompt", "").strip():
            error_msg = self.tr("请输入编辑提示词")
            return error_msg, task_params
        task_params["prompt"] = params["prompt"].strip()
        if "input_path" not in params or not params["input_path"]:
            error_msg = self.tr("请选择参考图")
            return error_msg, task_params
        if isinstance(params["input_path"], str) and " " in params["input_path"]:
            error_msg = self.tr("输入路径不能包含空格")
            return error_msg, task_params
        if isinstance(params["input_path"], str) and not params["input_path"].isascii():
            error_msg = self.tr("输入路径: 不支持非英文路径")
            return error_msg, task_params
        task_params["input_path"] = params["input_path"]
        if "model_name" not in params or not params["model_name"]:
            error_msg = self.tr("请选择编辑模型")
            return error_msg, task_params
        task_params["model_name"] = params["model_name"]
        if "mask_boxes" in params and params["mask_boxes"]:
            task_params["mask_boxes"] = params["mask_boxes"]
        if "output_path" not in params or not params["output_path"]:
            error_msg = self.tr("请设置文件保存位置")
            return error_msg, task_params
        if isinstance(params["output_path"], str) and not params["output_path"].isascii():
            error_msg = self.tr("输出路径: 不支持非英文路径")
            return error_msg, task_params
        task_params["output_path"] = params["output_path"]
        return error_msg, task_params


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
