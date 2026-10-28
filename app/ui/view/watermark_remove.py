import os
import sys
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (
    QVBoxLayout, QWidget, QLabel, QHBoxLayout, QStackedWidget, QLineEdit, QFileDialog, QStackedLayout, 
    QSlider, QButtonGroup, QPushButton, QFrame, QTextEdit
)
from PySide6.QtGui import QFont, QColor, QAction

from app.ui.library.qfluentwidgets import( 
    setFont, HeaderCardWidget, SegmentedWidget, ScrollArea, PushButton, CaptionLabel,
    LineEdit, FluentIcon, ComboBox, FlowLayout, TeachingTip, InfoBarIcon, TeachingTipTailPosition,
    MessageBox
)

from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.file_selector_widget import FileSelectorWidget
from app.ui.widgets.directory_selector_widget import DirectorySelectorWidget
from app.ui.widgets.custom_card_group_widget import StyleCard
from app.ui.widgets.image_preview_widget import SyncImageViewer, ImageNavigationWidget
from app.ui.widgets.video_preview_widget import SyncVideoViewer
from app.ui.widgets.status_bar_widget import StatusInfoWidget
from app.ui.widgets.task_info_messagebox_widget import TaskInfoMessageBox
from app.ui.widgets.custom_navigation import CustomNavigation
from app.ui.widgets.watermark_detect_settings import WatermarkDetectSettings

from app.ui.common.event_bus import global_event_bus
from app.controllers.task_manager import global_task_manager
from app.ui.common.task_params import bind_widget_to_param, LayeredTaskParams
from app.ui.common.task_status import TaskStatusModel
from app.ui.common.utils import get_file_type, global_backend_info_cache
from app.ui.common.config import cfg

from app.workers.watermark_remove import WatermarkRemoveWork, WatermarkRemoveUIParams

from app.license.globals import feature_gate


watermark_remove_params = LayeredTaskParams(params=WatermarkRemoveUIParams())
watermark_remove_task_status_model = TaskStatusModel()
# 当前批次已提交的任务 future，用于支持一次性取消
watermark_remove_active_futures = []

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
        bind_widget_to_param(singleFileSelector, "item_selected", watermark_remove_params, "io.input_path", transform=None)
        batchFilesSelector = DirectorySelectorWidget(self)
        batchFilesSelector.item_selected.connect(lambda file_path: global_event_bus.watermarkRemove_InputFileUpdate.emit(file_path))
        bind_widget_to_param(batchFilesSelector, "item_selected", watermark_remove_params, "io.input_path", transform=None)

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
        self.setTitle(self.tr("🔍 水印检测设置"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        watermark_detect_settings = WatermarkDetectSettings(parent=self)
        bind_widget_to_param(watermark_detect_settings, "watermarkDetectType", watermark_remove_params, "detect.watermark_detect_type", transform=None)
        bind_widget_to_param(watermark_detect_settings, "videoWatermarkFormat", watermark_remove_params, "detect.video_watermark_format", transform=None)
        bind_widget_to_param(watermark_detect_settings, "videoWatermarkAIDetectStrategy", watermark_remove_params, "detect.video_watermark_ai_detect_strategy", transform=None)
        bind_widget_to_param(watermark_detect_settings, "videoWatermarkAIDetectFrameRange", watermark_remove_params, "detect.video_watermark_ai_detect_frame_range", transform=None)
        bind_widget_to_param(watermark_detect_settings, "videoWatermarkAIDetectAnchorFrame", watermark_remove_params, "detect.video_watermark_ai_detect_anchor_frame", transform=None)
        bind_widget_to_param(watermark_detect_settings, "manualWatermarktMaskPath", watermark_remove_params, "detect.manual_watermark_mask_path", transform=None)
        bind_widget_to_param(watermark_detect_settings, "maskDirectoryChanged", watermark_remove_params, "detect.manual_watermark_mask_path", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkAIDetectMethod", watermark_remove_params, "detect.watermark_ai_detect_method", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkAIDetectPrompt", watermark_remove_params, "detect.watermark_ai_detect_prompt", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkAIDetectPosBoxes", watermark_remove_params, "detect.watermark_ai_detect_pos_boxes", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkAIDetectNegBoxes", watermark_remove_params, "detect.watermark_ai_detect_neg_boxes", transform=None)
        bind_widget_to_param(watermark_detect_settings, "watermarkAIDetectConfidence", watermark_remove_params, "detect.watermark_ai_detect_confidence", transform=None)
        bind_widget_to_param(watermark_detect_settings, "videoWatermarkTrackingData", watermark_remove_params, "detect.video_watermark_tracking_data", transform=None)

        global_event_bus.watermarkRemove_InputFileUpdate.connect(
            lambda file_path: (
                watermark_detect_settings.set_file_path(file_path=file_path),
                None if file_path else watermark_detect_settings.watermarkAIDetectPosBoxes.emit([]),
                None if file_path else watermark_detect_settings.watermarkAIDetectNegBoxes.emit([]),
                None if file_path else watermark_detect_settings.videoWatermarkTrackingData.emit({"keyframes": [], "end_frame": None, "reverse_tracking_frames": 8})
            )
        )

        main_layout.addWidget(watermark_detect_settings)


class WatermarkTypeCardItem(QFrame):
    clicked = Signal(str)

    def __init__(self, key, icon, name, description, color, parent=None):
        super().__init__(parent)
        self.key = key
        self.is_selected = False
        self.setFixedSize(95, 40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(description)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        icon_label = QLabel(icon, self)
        icon_label.setFixedSize(20, 20)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(
            f"QLabel {{ background-color: {color}; border-radius: 8px; font-size: 14px; }}"
        )
        layout.addWidget(icon_label)
        name_label = QLabel(name, self)
        setFont(name_label, 11)
        name_label.setStyleSheet("color: #1f2937; background: transparent;")
        layout.addWidget(name_label)
        self.set_selected(False)

    def set_selected(self, selected):
        self.is_selected = bool(selected)
        if self.is_selected:
            self.setStyleSheet(
                "WatermarkTypeCardItem { background: #eef2ff; border: 2px solid #667eea; border-radius: 8px; }"
            )
        else:
            self.setStyleSheet("""
                WatermarkTypeCardItem { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; } 
                WatermarkTypeCardItem:hover { background: #f8fafc; border-color: #c7d2fe; }
            """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key)
            event.accept()
            return
        super().mousePressEvent(event)

class WatermarkRemoveStyleCard(HeaderCardWidget):
    model_name = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🎨 水印去除设置"))
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
        self.tab_image = QPushButton(self.tr("通用图片"))
        setFont(self.tab_image, fontSize=14, weight=QFont.DemiBold)
        self.tab_image.setCheckable(True)
        self.tab_image.setChecked(True)
        self.tab_document = QPushButton(self.tr("文档图片"))
        setFont(self.tab_document, fontSize=14, weight=QFont.DemiBold)
        self.tab_document.setCheckable(True)
        self.tab_video = QPushButton(self.tr("视频"))
        setFont(self.tab_video, fontSize=14, weight=QFont.DemiBold)
        self.tab_video.setCheckable(True)
        self.tab_group.addButton(self.tab_image, 0)
        self.tab_group.addButton(self.tab_document, 1)
        self.tab_group.addButton(self.tab_video, 2)
        tab_layout.addWidget(self.tab_image)
        tab_layout.addWidget(self.tab_document)
        tab_layout.addWidget(self.tab_video)
        main_layout.addWidget(tab_widget)

        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        self.all_cards: list[StyleCard] = []

        self.image_container = QWidget()
        image_outer_layout = QVBoxLayout(self.image_container)
        image_outer_layout.setContentsMargins(0, 6, 0, 6)
        image_outer_layout.setSpacing(8)

        self.image_cards_widget = QWidget()
        sp = self.image_cards_widget.sizePolicy()
        sp.setHeightForWidth(True)
        self.image_cards_widget.setSizePolicy(sp)
        image_layout = FlowLayout(self.image_cards_widget, needAni=True, isTight=True)
        image_layout.setContentsMargins(0, 0, 0, 4)
        image_layout.setHorizontalSpacing(8)
        image_layout.setVerticalSpacing(8)

        general_edit_card = StyleCard("#667eea", self.tr("智能重绘"), self.tr("复杂场景，效果好"), vertical_layout=True)
        general_edit_card.set_name("general_edit")
        image_layout.addWidget(general_edit_card)

        object_removal_card = StyleCard("#e3fb93", self.tr("智能消除"), self.tr("精准消除，速度快"), vertical_layout=True)
        object_removal_card.set_name("object_removal")
        image_layout.addWidget(object_removal_card)

        patchwiper_card = StyleCard("#4facfe", self.tr("细节增强"), self.tr("细节重建，不适合字幕"), vertical_layout=True)
        patchwiper_card.set_name("patchwiper")
        image_layout.addWidget(patchwiper_card)

        coordfill_card = StyleCard("#fbc2eb", self.tr("快速填充"), self.tr("简单背景，极速处理"), vertical_layout=True)
        coordfill_card.set_name("coordfill")
        image_layout.addWidget(coordfill_card)

        lama_card = StyleCard("#84fab0", self.tr("自然保守"), self.tr("自然衔接，保留不足"), vertical_layout=True)
        lama_card.set_name("lama")
        image_layout.addWidget(lama_card)

        emdf_card = StyleCard("#f093fb", self.tr("自适应修补"), self.tr("稳定修补，细节易失"), vertical_layout=True)
        emdf_card.set_name("emdf")
        image_layout.addWidget(emdf_card)

        grig_card = StyleCard("#a18cd1", self.tr("平衡修复"), self.tr("结构稳定，处理快"), vertical_layout=True)
        grig_card.set_name("grig")
        image_layout.addWidget(grig_card)

        self.image_cards = [
            general_edit_card, 
            object_removal_card,
            patchwiper_card,
            coordfill_card,
            lama_card,  
            emdf_card, 
            grig_card
        ]
        self.image_model_visible_count = 4
        self.image_models_expanded = False

        self.image_more_button = QPushButton(self.tr("展开更多模型 ▾"), self.image_container)
        self.image_more_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image_more_button.setFixedHeight(32)
        self.image_more_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #667eea;
                border: none;
                border-radius: 6px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background: #f1f3ff;
            }
        """)
        self.image_more_button.clicked.connect(self.toggle_image_models)

        image_model_card = QFrame()
        image_model_card.setObjectName("modelCard")
        image_model_card.setStyleSheet("""
            QFrame#modelCard {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
        """)
        image_model_card_layout = QVBoxLayout(image_model_card)
        image_model_card_layout.setContentsMargins(12, 10, 12, 12)
        image_model_card_layout.setSpacing(8)
        image_model_card_layout.addWidget(self.image_cards_widget)
        image_model_card_layout.addWidget(self.image_more_button)
        image_outer_layout.addWidget(image_model_card)

        self.image_params_widget = self._build_mask_dilate_params()
        image_outer_layout.addWidget(self.image_params_widget)

        self.stacked_widget.addWidget(self.image_container)

        self._set_extra_image_models_visible(False)

        self.all_cards.extend(self.image_cards)

        document_container = QWidget()
        document_outer_layout = QVBoxLayout(document_container)
        document_outer_layout.setContentsMargins(0, 6, 0, 6)
        document_outer_layout.setSpacing(8)

        document_cards_widget = QWidget()
        document_flow = FlowLayout(document_cards_widget, isTight=True)
        document_flow.setContentsMargins(0, 0, 0, 4)
        document_flow.setHorizontalSpacing(8)
        document_flow.setVerticalSpacing(8)

        image_restoration_card = StyleCard("#f093fb", self.tr("智能修补"), self.tr("文档截图，精细修复"), vertical_layout=True)
        image_restoration_card.set_name("image_restoration")
        document_flow.addWidget(image_restoration_card)

        document_model_card = QFrame()
        document_model_card.setObjectName("docModelCard")
        document_model_card.setStyleSheet("""
            QFrame#docModelCard {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
        """)
        document_model_card_layout = QVBoxLayout(document_model_card)
        document_model_card_layout.setContentsMargins(12, 10, 12, 12)
        document_model_card_layout.setSpacing(8)
        document_model_card_layout.addWidget(document_cards_widget)
        document_outer_layout.addWidget(document_model_card)

        document_params_widget = self._build_restoration_prompt_params()
        document_outer_layout.addWidget(document_params_widget)

        self.stacked_widget.addWidget(document_container)

        self.document_cards = [image_restoration_card]
        self.all_cards.extend(self.document_cards)

        video_container = QWidget()
        video_outer_layout = QVBoxLayout(video_container)
        video_outer_layout.setContentsMargins(0, 6, 0, 6)
        video_outer_layout.setSpacing(8)

        video_cards_widget = QWidget()
        video_flow = FlowLayout(video_cards_widget, isTight=True)
        video_flow.setContentsMargins(0, 0, 0, 4)
        video_flow.setHorizontalSpacing(8)
        video_flow.setVerticalSpacing(8)

        video_engine_card = StyleCard("#2980b9", self.tr("关键帧擦除"), self.tr("智能传播，速度适中"), vertical_layout=True)
        video_engine_card.set_name("video_engine")
        video_flow.addWidget(video_engine_card)

        video_seq_card = StyleCard("#9b59b6", self.tr("时序擦除"), self.tr("跨帧追踪，减少闪烁"), vertical_layout=True)
        video_seq_card.set_name("ppt")
        video_flow.addWidget(video_seq_card)

        video_model_card = QFrame()
        video_model_card.setObjectName("videoModelCard")
        video_model_card.setStyleSheet("""
            QFrame#videoModelCard {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
        """)
        video_model_card_layout = QVBoxLayout(video_model_card)
        video_model_card_layout.setContentsMargins(12, 10, 12, 12)
        video_model_card_layout.setSpacing(8)
        video_model_card_layout.addWidget(video_cards_widget)
        video_outer_layout.addWidget(video_model_card)

        video_params_widget = self._build_mask_dilate_params()
        video_outer_layout.addWidget(video_params_widget)

        self.stacked_widget.addWidget(video_container)
        
        self.video_cards = [video_engine_card, video_seq_card]
        self.all_cards.extend(self.video_cards)

        self.tab_image.toggled.connect(lambda checked: self.on_tab_changed(0, checked))
        self.tab_document.toggled.connect(lambda checked: self.on_tab_changed(1, checked))
        self.tab_video.toggled.connect(lambda checked: self.on_tab_changed(2, checked))

        for card in self.all_cards:
            card.mousePressEvent = lambda event, c=card: self.on_card_clicked(c)

        bind_widget_to_param(self, "model_name", watermark_remove_params, "model.model_name", transform=None)
        self.select_first_interactive()
        global_event_bus.watermarkRemove_InputFileUpdate.connect(lambda file_path: self.update_default_models(file_path=file_path))
        global_event_bus.License_update.connect(self.select_first_interactive)
        global_event_bus.watermarkRemove_TaskFinishedByModel.connect(self.update_model_card_info)

    def showEvent(self, event):
        super().showEvent(event)
        self._update_stacked_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_stacked_height()

    def _update_stacked_height(self):
        current_index = self.stacked_widget.currentIndex()
        page_widget = self.stacked_widget.widget(current_index)
        if page_widget:
            available_width = self.stacked_widget.width()
            if available_width <= 0:
                available_width = self.width() - 20
            page_h = self._calc_widget_height(page_widget, available_width)
            self.stacked_widget.setFixedHeight(page_h)

    def _card_pool(self, index: int) -> list:
        return {0: self.image_cards, 1: self.document_cards, 2: self.video_cards}.get(index, self.image_cards)

    def _build_params_section_header(self) -> QWidget:
        header = QWidget()
        header.setStyleSheet("QWidget { background: transparent; }")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(0, 4, 0, 0)
        h_layout.setSpacing(8)

        icon_label = QLabel("⚙️")
        setFont(icon_label, 13)
        icon_label.setStyleSheet("color: #555555; background: transparent;")

        title_label = QLabel(self.tr("参数设置"))
        setFont(title_label, 13, QFont.DemiBold)
        title_label.setStyleSheet("color: #555555; background: transparent;")

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #e2e8f0;")

        h_layout.addWidget(icon_label)
        h_layout.addWidget(title_label)
        h_layout.addWidget(line, 1)
        return header

    def _build_mask_dilate_params(self) -> QWidget:
        container = QFrame()
        container.setObjectName("paramsCard")
        container.setStyleSheet("""
            QFrame#paramsCard {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._build_params_section_header())

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        name_label = QLabel(self.tr("水印 Mask 扩张系数"))
        setFont(name_label, 13, QFont.DemiBold)
        name_label.setStyleSheet("color: #2c3e50; background: transparent;")
        hint_label = QLabel(self.tr("让遮罩稍外扩，盖住水印边缘"))
        setFont(hint_label, 10)
        hint_label.setStyleSheet("color: #888888; background: transparent;")
        value_label = QLabel("2")
        value_label.setFixedWidth(20)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        setFont(value_label, 13, QFont.DemiBold)
        value_label.setStyleSheet("color: #3498db; background: transparent;")
        row.addWidget(name_label)
        row.addWidget(hint_label, 1)
        row.addWidget(value_label)
        layout.addLayout(row)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(8)
        slider.setValue(2)
        slider.setTickInterval(1)
        slider.setSingleStep(1)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e2e8f0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: none;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #3498db;
                border-radius: 3px;
            }
        """)
        slider.setFixedHeight(22)
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        bind_widget_to_param(slider, "valueChanged", watermark_remove_params, "model.mask_dilate", transform=None)
        slider.valueChanged.emit(2)
        layout.addWidget(slider)
        return container

    def _build_restoration_prompt_params(self) -> QWidget:
        PRESETS = (
            ("general",    "💧", self.tr("通用水印"),   "#38bdf8"),
            ("horizontal", "↔️", self.tr("半透明水平"), "#22c55e"),
            ("vertical",   "↕️", self.tr("半透明垂直"), "#f59e0b"),
            ("diagonal",   "📐", self.tr("半透明斜向"), "#f472b6"),
            ("logo",       "©️", self.tr("图片/Logo"),  "#0ea5e9"),
        )
        container = QFrame()
        container.setObjectName("restorationParamsCard")
        container.setStyleSheet("""
            QFrame#restorationParamsCard {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._build_params_section_header())

        type_label = QLabel(self.tr("水印类型"))
        setFont(type_label, 13, QFont.DemiBold)
        type_label.setStyleSheet("color: #2c3e50; background: transparent;")
        layout.addWidget(type_label)

        type_cards: list[WatermarkTypeCardItem] = []
        flow_widget = QWidget()
        flow_widget.setStyleSheet("QWidget { background: transparent; }")
        flow_layout = FlowLayout(flow_widget, needAni=True)
        flow_layout.setContentsMargins(0, 0, 0, 0)
        flow_layout.setHorizontalSpacing(8)
        flow_layout.setVerticalSpacing(8)

        def _on_type_selected(selected_key):
            for tc in type_cards:
                tc.set_selected(tc.key == selected_key)

        for key, icon, name, color in PRESETS:
            tc = WatermarkTypeCardItem(key, icon, name, name, color, container)
            tc.clicked.connect(_on_type_selected)
            bind_widget_to_param(tc, "clicked", watermark_remove_params, "model.restoration_watermark_type")
            flow_layout.addWidget(tc)
            type_cards.append(tc)
        layout.addWidget(flow_widget)

        prompt_label = QLabel(self.tr("自定义水印类型（可选）"))
        setFont(prompt_label, 13, QFont.DemiBold)
        prompt_label.setStyleSheet("color: #2c3e50; background: transparent;")
        layout.addWidget(prompt_label)

        prompt_edit = QTextEdit(container)
        prompt_edit.setMaximumHeight(72)
        prompt_edit.setPlaceholderText(self.tr("描述什么类型水印；留空则使用上方预设（建议使用英文）"))
        prompt_edit.setStyleSheet("""
            QTextEdit { border: 1px solid #e0e0e0; border-radius: 8px; padding: 6px;
                        background-color: #fafafa; color: #1a1a1a; }
            QTextEdit:focus { border: 1px solid #667eea; background-color: #ffffff; }
        """)
        setFont(prompt_edit, 13)
        bind_widget_to_param(prompt_edit, "textChanged", watermark_remove_params, "model.restoration_prompt", transform=lambda: prompt_edit.toPlainText())
        layout.addWidget(prompt_edit)

        type_cards[0].clicked.emit(type_cards[0].key)
        return container

    def _set_extra_image_models_visible(self, visible: bool):
        for card in self.image_cards[self.image_model_visible_count:]:
            card.setVisible(visible)

    def toggle_image_models(self):
        self.image_models_expanded = not self.image_models_expanded
        is_expanded = self.image_models_expanded
        self._set_extra_image_models_visible(is_expanded)

        self.image_more_button.setText(self.tr("收起模型 ▴") if is_expanded else self.tr("展开更多模型 ▾"))

        self.image_cards_widget.layout().invalidate()
        self.image_cards_widget.layout().activate()
        self._animate_to_content_height()

    def update_model_card_info(self, model_name):
        current_index = self.stacked_widget.currentIndex()
        active_pool = self._card_pool(current_index)
        for one_card in active_pool:
            if one_card.get_name() != model_name:
                continue
            one_card.UpdateLicenseInfo.emit()
            if not one_card.is_interactive():
                self.model_name.emit("")
                self.select_first_interactive()
            break

    def on_tab_changed(self, index, checked):
        if not checked:
            return
        self.stacked_widget.setCurrentIndex(index)
        target_pool = self._card_pool(index)
        if target_pool:
            self.on_card_clicked(target_pool[0])
        self._animate_to_content_height()

    def _animate_to_content_height(self):
        current_index = self.stacked_widget.currentIndex()
        page_widget = self.stacked_widget.widget(current_index)

        available_width = self.stacked_widget.width()
        if available_width <= 0:
            available_width = self.width() - 20

        page_h = self._calc_widget_height(page_widget, available_width)
        self.stacked_widget.setFixedHeight(page_h)

        self.viewLayout.invalidate()
        self.viewLayout.activate()
        target_height = self.sizeHint().height()
        self.animate_height_change(target_height)

    def _calc_widget_height(self, widget: QWidget, width: int) -> int:
        layout = widget.layout()
        if layout is None:
            h = widget.sizeHint().height()
        else:
            h = self._calc_layout_height(layout, width)
        min_h = widget.minimumHeight()
        max_h = widget.maximumHeight()
        if min_h > 0:
            h = max(h, min_h)
        if max_h < 16777215:
            h = min(h, max_h)
        return h

    def _calc_layout_height(self, layout, width: int) -> int:
        if isinstance(layout, FlowLayout):
            return layout.heightForWidth(width)

        m = layout.contentsMargins()
        inner_width = width - m.left() - m.right()
        total_h = m.top() + m.bottom()
        item_heights = []

        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                if not w.isVisible():
                    continue
                item_heights.append(self._calc_widget_height(w, inner_width))
            elif item.layout() is not None:
                item_heights.append(self._calc_layout_height(item.layout(), inner_width))
            else:
                sp = item.spacerItem()
                if sp is not None:
                    item_heights.append(sp.sizeHint().height())

        if isinstance(layout, QHBoxLayout):
            total_h += max(item_heights) if item_heights else 0
        else:
            total_h += sum(item_heights)
            if len(item_heights) > 1:
                total_h += layout.spacing() * (len(item_heights) - 1)

        return total_h

    def on_card_clicked(self, clicked_card):
        if not clicked_card.is_interactive():
            return
        current_index = self.stacked_widget.currentIndex()
        active_pool = self._card_pool(current_index)
        for c in active_pool:
            c.set_selected(False)
        clicked_card.set_selected(True)
        self.model_name.emit(clicked_card.get_name())

    def select_first_interactive(self):
        current_index = self.stacked_widget.currentIndex()
        active_pool = self._card_pool(current_index)
        if any(c.is_selected for c in active_pool):
            return
        for card in active_pool:
            if card.is_interactive():
                card.set_selected(True)
                self.model_name.emit(card.get_name())
                return
        self.model_name.emit("")

    def update_default_models(self, file_path):
        if not file_path:
            self.tab_image.setEnabled(True)
            self.tab_document.setEnabled(True)
            self.tab_video.setEnabled(True)
            self.tab_image.setChecked(True)
            return
        if os.path.isfile(file_path):
            file_type = get_file_type(file_path)
        else:
            file_type = get_file_type(os.path.join(file_path, os.listdir(file_path)[0]))
        if file_type == "image":
            self.tab_image.setEnabled(True)
            self.tab_document.setEnabled(True)
            self.tab_video.setEnabled(False)
            self.tab_image.setChecked(True)
            self.on_tab_changed(index=0, checked=True)
        elif file_type == "video":
            self.tab_image.setEnabled(True)
            self.tab_document.setEnabled(True)
            self.tab_video.setEnabled(True)
            self.tab_video.setChecked(True)
            self.on_tab_changed(index=2, checked=True)
        else:
            self.tab_image.setEnabled(True)
            self.tab_document.setEnabled(True)
            self.tab_video.setEnabled(True)
            self.tab_image.setChecked(True)

    def animate_height_change(self, target_height: int):
        start_height = self.height()
        self.setMinimumHeight(0)
        self.height_anim = QPropertyAnimation(self, b"maximumHeight")
        self.height_anim.setDuration(250)
        self.height_anim.setStartValue(start_height)
        self.height_anim.setEndValue(target_height)
        self.height_anim.setEasingCurve(QEasingCurve.OutCubic)
        # 动画结束后解除 maximumHeight 限制，允许父布局自由调整
        self.height_anim.finished.connect(lambda: self.setMaximumHeight(16777215))
        self.height_anim.start()

        
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
        bind_widget_to_param(self.save_location_line_edit, "textChanged", watermark_remove_params, "io.output_path", transform=None)
        output_settings_layout.addWidget(self.save_location_line_edit)
        output_settings_layout.addSpacing(10)

        output_format_label = CaptionLabel(text=self.tr("输出格式"))
        setFont(output_format_label, 13)
        output_format_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        output_settings_layout.addWidget(output_format_label)
        self.output_format_combo = ComboBox()
        self.output_format_combo.addItems([self.tr("保持原格式")])
        bind_widget_to_param(self.output_format_combo, "currentTextChanged", watermark_remove_params, "io.output_format", transform=None)
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
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0)
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

        self.output_mask_cards: list[HeaderCardWidget] = []

        fileSelectorCard = FileSelectorCard(self)
        main_layout.addWidget(fileSelectorCard)

        watermarkRemoveStyleCard = WatermarkRemoveStyleCard(self)
        main_layout.addWidget(watermarkRemoveStyleCard)

        watermarkDetectionTypeCard = WatermarkDetectionTypeCard(self)
        main_layout.addWidget(watermarkDetectionTypeCard)
        self.output_mask_cards.append(watermarkDetectionTypeCard)

        outputSettingsCard = OutputSettingsCard(self)
        main_layout.addWidget(outputSettingsCard)

        self.setWidget(view)
        self.setViewportMargins(0, 0, 0, 0)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_layout.addStretch(1)

        watermark_remove_params.param_changed.connect(self._on_param_changed)
        self._on_param_changed("model.model_name", watermark_remove_params.get_param("model.model_name"))
        
    def _on_param_changed(self, key, value):
        if key != "model.model_name":
            return
        if value == "image_restoration":
            for card in self.output_mask_cards:
                card.hide()
        else:
            for card in self.output_mask_cards:
                card.setVisible(True)



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
        self.status_info_widget = StatusInfoWidget(watermark_remove_task_status_model, self)
        self.status_info_widget.model.set_pipeline_steps(names=[self.tr('准备任务'), self.tr('检测水印'), self.tr('去除水印'), self.tr('导出文件')])
        self.status_info_widget.cancel_requested.connect(self._cancel_tasks)
        bottom_layout.addWidget(self.status_info_widget, 2)

        main_layout.addLayout(bottom_layout)

        self.files_preview_info = {}
        self.media_type = "image"

        global_event_bus.watermarkRemove_InputFileUpdate.connect(self.update_init_preview)
        global_event_bus.watermarkRemove_TaskFinished.connect(self.update_preview)
        global_event_bus.watermarkRemove_TaskProgress.connect(self.update_process)
        global_event_bus.watermarkRemove_PreviewFile.connect(self._on_preview_file)
        global_event_bus.watermarkRemove_ImageNavigationInit.connect(lambda: self.image_navigation_widget.clear_images())

        watermark_remove_params.param_changed.connect(self._on_param_changed)
                
    def _on_param_changed(self, key, value):
        if key != "model.model_name":
            return
        if value in ["image_restoration", "video_engine"]:
            self.status_info_widget.model.set_pipeline_steps(names=[self.tr('准备任务'), self.tr('去除水印'), self.tr('导出文件')])
            self.navigation.set_item_visible(index=1, visible=False)
        else:
            self.status_info_widget.model.set_pipeline_steps(names=[self.tr('准备任务'), self.tr('检测水印'), self.tr('去除水印'), self.tr('导出文件')])
            self.navigation.set_item_visible(index=1, visible=True)

    def update_init_preview(self, file_path):
        self.image_navigation_widget.clear_images()
        self.image_viewer.init_scene()
        self.video_viewer.init_scene()
        self.files_preview_info = {}
        self.media_type = "image"
        if not file_path:
            self.navigation.hide()
            self.stack.setCurrentIndex(0)
            watermark_remove_task_status_model.reset()
            return
        self.navigation.show()
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
        elif file_type == "video":
            self.stack.setCurrentIndex(2)
            self.media_type = "video"
        else:
            self.navigation.hide()
            self.placeholder_widget.setText(f"不支持的文件类型: {ext}")
            self.stack.setCurrentIndex(0)
            watermark_remove_task_status_model.reset()

    def update_preview(self, input_path, output_path):
        preview_mode = "ResultCompared"
        if preview_mode not in self.files_preview_info:
            self.files_preview_info[preview_mode] = {}
        self.files_preview_info[preview_mode][input_path] = output_path
        current_file_path = self.image_navigation_widget.get_current_image()
        if current_file_path:
            self._on_preview_file(current_file_path)
        else:
            self.image_navigation_widget.load_images([input_path], self.media_type)

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

    def _cancel_tasks(self):
        cancelled = 0
        for future in list(watermark_remove_active_futures):
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
        error_msg = self._params_check(params=watermark_remove_params)
        if error_msg:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        w = TaskInfoMessageBox(watermark_remove_params, "watermark-remove", self.window())
        if not w.exec():
            return
        
        allowed_use, error_msg = feature_gate.can_use(feature_name=feature_gate.get_feature_name(watermark_remove_params.get_param("model.model_name")), return_errmsg=True)
        if not allowed_use:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        watermark_remove_params.params._feature_name_ = feature_gate.get_feature_name(watermark_remove_params.get_param("model.model_name"))

        total_tasks = []
        input_path = watermark_remove_params.get_param("io.input_path")
        global_event_bus.watermarkRemove_ImageNavigationInit.emit()
        if os.path.isdir(input_path):
            self.is_batch_task = True
            for one_file in os.listdir(input_path):
                watermark_remove_params.set_param("io.input_path", os.path.join(input_path, one_file))
                task_instance = WatermarkRemoveWork(watermark_remove_params)
                func, args, kwargs = task_instance.to_worker()
                total_tasks.append((func, args, kwargs))
        else:
            self.is_batch_task = False
            task_instance = WatermarkRemoveWork(watermark_remove_params)
            func, args, kwargs = task_instance.to_worker()
            total_tasks.append((func, args, kwargs))

        backend_type, gpu_name = global_backend_info_cache.get(key="backend_info")
        watermark_remove_task_status_model.start_batch(total=len(total_tasks), backend_type=backend_type, gpu_name=gpu_name)
        if not self.is_batch_task:
            watermark_remove_task_status_model.start_step(name=self.tr("准备任务"))

        watermark_remove_active_futures.clear()

        def submit_task(func, args, kwargs, input_path):
            retry_callback = lambda f=func, a=args, k=kwargs, p=input_path: submit_task(f, a, k, p)
            future = global_task_manager.submit(func, *args, **kwargs)
            watermark_remove_active_futures.append(future)
            future.finished.connect(lambda result, path=input_path: self._task_finished(path, result))
            future.failed.connect(
                lambda error, path=input_path, retry=retry_callback: watermark_remove_task_status_model.report_failure(path, error, retry)
            )
            future.cancelled.connect(
                lambda path=input_path, retry=retry_callback: watermark_remove_task_status_model.report_failure(path, "任务被取消", retry)
            )
            future.progress.connect(lambda value, msg, path=input_path: self._task_progress(path, value, msg))

        for func, args, kwargs in total_tasks:
            input_path = args[1].get_param("io.input_path")
            submit_task(func, args, kwargs, input_path)
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
        if value == "MaskCompleted":
            global_event_bus.watermarkRemove_TaskProgress.emit(input_path, value, msg)
            if not self.is_batch_task:
                watermark_remove_task_status_model.finish_step(self.tr('检测水印'))
                watermark_remove_task_status_model.start_step(self.tr('去除水印'))
        elif value == "MaskStart":
            if not self.is_batch_task:
                watermark_remove_task_status_model.finish_step(self.tr("准备任务"))
                watermark_remove_task_status_model.start_step(self.tr('检测水印'))
        elif value == "WaterRemoveStart":
            if not self.is_batch_task:
                watermark_remove_task_status_model.finish_step(self.tr("准备任务"))
                watermark_remove_task_status_model.start_step(self.tr('去除水印'))
        elif value == "WaterRemoved":
            if not self.is_batch_task:
                watermark_remove_task_status_model.finish_step(self.tr("去除水印"))
                watermark_remove_task_status_model.start_step(self.tr('导出文件'))

    def _task_finished(self, input_path, output_path):
        if not feature_gate.is_pro:
            try:
                feature_gate.use_feature(feature_name=feature_gate.get_feature_name(watermark_remove_params.get_param("model.model_name")))
            except Exception:
                pass
            global_event_bus.watermarkRemove_TaskFinishedByModel.emit(watermark_remove_params.get_param("model.model_name"))
        watermark_remove_task_status_model.report_success()
        global_event_bus.watermarkRemove_TaskFinished.emit(input_path, output_path)
        if not self.is_batch_task:
            watermark_remove_task_status_model.finish_step(name=self.tr("导出文件"))

    def _params_check(self, params: LayeredTaskParams):
        error_msg = ""
        if not cfg.get(cfg.localWatermarkRemoveEnabled):
            error_msg = self.tr("请在设置页面打开 '水印移除' 能力开关")
            return error_msg
        if get_file_type(params.get_param("io.input_path")) == "video" and \
           cfg.get(cfg.additionalParams)["SoftwareSettings"]["FFmpeg_status_info"]["text"] != "OK":
            error_msg = self.tr("请在设置页面配置软件 FFmpeg 的正确路径并通过验证")
            return error_msg
        error_msg = params.params.validate()
        return error_msg


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
        