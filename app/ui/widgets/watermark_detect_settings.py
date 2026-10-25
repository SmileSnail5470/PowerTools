import os
import sys
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QFileDialog, QLabel, QPushButton, QLineEdit,
    QStackedWidget, QSpinBox, QFrame, QGraphicsDropShadowEffect, QDialog
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Signal, QTimer, QSize
from PySide6.QtGui import QColor, QFont, QImage
from app.ui.library.qfluentwidgets import setFont, MessageBox, FlowLayout, FluentIcon
from app.ui.common.utils import get_file_type
from app.ui.common.config import cfg
from app.ui.widgets.watermark_interactive_widget import AreaSelectorDialog
from app.ui.widgets.watermark_manual_select_widget import WatermarkMaskTool
from app.ui.widgets.video_object_detector_widget import VideoFrameCanvas, VideoObjectDetectorWidget
from app.utils.image_io import open_video_capture


class WatermarkVideoPreviewDialog(QDialog):
    def __init__(self, file_path, total_frames, initial_frame=1, parent=None):
        super().__init__(parent)
        self.capture = open_video_capture(file_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) if self.capture.isOpened() else 0
        if not self.fps or self.fps <= 0:
            self.fps = 30

        self.setWindowTitle(self.tr("预览视频"))
        self.resize(900, 600)

        layout = QVBoxLayout(self)
        self.canvas = VideoFrameCanvas(self, need_hover_hint=False)
        self.canvas.set_view_mode("original")
        self.canvas.setMinimumSize(480, 320)
        layout.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        self.play_button = QPushButton(self.tr("播放"))
        self.play_button.setObjectName("secondaryBtn")
        self.play_button.setFixedHeight(34)
        self.play_button.clicked.connect(self._toggle_playback)
        controls.addWidget(self.play_button)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, max(1, total_frames))
        self.frame_slider.setValue(min(max(1, initial_frame), max(1, total_frames)))
        self.frame_slider.valueChanged.connect(self._show_frame)
        controls.addWidget(self.frame_slider, 1)

        self.frame_label = QLabel()
        controls.addWidget(self.frame_label)
        layout.addLayout(controls)

        self._show_frame(self.frame_slider.value())

    def _show_frame(self, frame_number):
        if not self.capture.isOpened():
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        success, frame = self.capture.read()
        if success:
            self._display_frame(frame)
        else:
            self._show_frame_error()
        self.frame_label.setText(self.tr("{0} / {1} 帧").format(frame_number, self.frame_slider.maximum()))

    def _display_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        self.canvas.set_frame(image, show_detection_overlay=False, show_prompts=False)

    def _show_frame_error(self):
        self.canvas.set_frame(
            QImage(),
            placeholder=self.tr("无法读取视频帧"),
            show_detection_overlay=False,
            show_prompts=False,
        )

    def _toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText(self.tr("播放"))
            return
        if self.frame_slider.value() >= self.frame_slider.maximum():
            self.frame_slider.setValue(1)
        self.timer.start(max(1, round(1000 / self.fps)))
        self.play_button.setText(self.tr("暂停"))

    def _advance_frame(self):
        if self.frame_slider.value() >= self.frame_slider.maximum():
            self.timer.stop()
            self.play_button.setText(self.tr("播放"))
            return
        frame_number = self.frame_slider.value() + 1
        success, frame = self.capture.read()
        if not success:
            self.timer.stop()
            self.play_button.setText(self.tr("播放"))
            self._show_frame_error()
            return
        previous_block_state = self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_number)
        self.frame_slider.blockSignals(previous_block_state)
        self._display_frame(frame)
        self.frame_label.setText(self.tr("{0} / {1} 帧").format(frame_number, self.frame_slider.maximum()))

    def closeEvent(self, event):
        self.timer.stop()
        self.capture.release()
        super().closeEvent(event)


class AutoSizeStackedWidget(QStackedWidget):
    def sizeHint(self):
        w = self.currentWidget()
        if w is not None:
            return w.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        if w is not None:
            return w.minimumSizeHint()
        return super().minimumSizeHint()


class OptionCard(QFrame):
    clicked = Signal()

    def __init__(self, title, desc, parent=None, icon=None, icon_color="#DBEAFE", recommended=False):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.selected = False
        
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("optTitle")
        setFont(self.title_label, 11 if icon else 13, QFont.Medium)
        if icon:
            header = QHBoxLayout()
            header.setSpacing(4)
            self.icon_label = QLabel(icon)
            self.icon_label.setFixedSize(20, 20)
            self.icon_label.setAlignment(Qt.AlignCenter)
            self.icon_label.setStyleSheet(f"background-color: {icon_color}; border-radius: 7px; font-size: 13px;")
            header.addWidget(self.icon_label)
            header.addStretch()
            if recommended:
                self.recommended_label = QLabel(self.tr("推荐"))
                self.recommended_label.setObjectName("recommendBadge")
                setFont(self.recommended_label, 8, QFont.DemiBold)
                header.addWidget(self.recommended_label)
            layout.addLayout(header)
            layout.addWidget(self.title_label)
        else:
            layout.addWidget(self.title_label)
        self.desc_label = QLabel(desc)
        self.desc_label.setObjectName("optDesc")
        self.desc_label.setWordWrap(True)
        setFont(self.desc_label, 9 if icon else 11)
        if desc:
            layout.addWidget(self.desc_label)
        layout.setContentsMargins(7 if icon else 14, 6 if icon else 14, 7 if icon else 14, 6 if icon else 14)
        layout.setSpacing(2 if icon else 2)
        self.update_style()

    def update_style(self):
        self.setProperty("selected", self.selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.title_label.style().unpolish(self.title_label)
        self.title_label.style().polish(self.title_label)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()

class WatermarkDetectSettings(QWidget):
    watermarkDetectType = Signal(str)         # AI检测/人工检测
    watermarkFormat = Signal(str)             # 静态水印/动态水印
    manualWatermarktMaskPath = Signal(str)    # 手动标注 Mask 路径
    maskDirectoryChanged = Signal(str)        # 人工指定 Mask 路径
    imageBoxes = Signal(list)                 # 人工选中输入区域
    # AI 交互检测信号
    watermarkAIInteractiveType = Signal(str)  # 语义检测/空间范围检测
    watermarkDetectPrompt = Signal(str)       # 语义检测提示词
    watermarkBoxes = Signal(list)             # 水印空间范围 boxes 坐标
    watermarkConfidence = Signal(float)       # 水印置信度
    # AI 自动检测
    watermarkContent = Signal(str)            # 通用水印/文字水印
    # 动态水印跟踪信号
    watermarkTrackingData = Signal(dict)      # 跟踪数据: {keyframes, end_frame, tracking_enabled, reverse_tracking_frames}


    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.current_tab_index = 0
        self.file_path = None
        self.mask_dir_path = ""
        self.is_video_file = False
        self.is_dynamic_watermark = False
        self.video_detection_strategy = "frame_by_frame"
        self.total_video_frames = 1

        self.setObjectName("WatermarkDetectSettings")
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame()
        self.card.setObjectName("panelCard")
        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setContentsMargins(0, 0, 0, 0)
        self.card_layout.setSpacing(0)

        self.tabs_container = QFrame()
        self.tabs_container.setObjectName("tabsContainer")
        self.tabs_container.setFixedHeight(46)
        self.tabs_layout = QHBoxLayout(self.tabs_container)
        self.tabs_layout.setContentsMargins(5, 5, 5, 5)
        self.tabs_layout.setSpacing(0)
        
        self.tab_btns = []
        for i, text in enumerate((self.tr("AI检测"), self.tr("手工标注"))):
            btn = QPushButton(text)
            btn.setObjectName("tabItem")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            setFont(btn, 13, QFont.DemiBold)
            btn.clicked.connect(lambda _, idx=i: self.switch_tab(idx))
            self.tabs_layout.addWidget(btn)
            self.tab_btns.append(btn)
            if i == 0:
                btn.setChecked(True)

        self.card_layout.addWidget(self.tabs_container)
        self.card_layout.addSpacing(8)

        self.stack = QStackedWidget()
        ai_page = QWidget()
        ai_layout = QVBoxLayout(ai_page)
        ai_layout.setContentsMargins(0, 0, 0, 0)
        ai_layout.setSpacing(12)

        self.detection_types_card = QFrame()
        self.detection_types_card.setObjectName("sectionCard")
        detection_types_layout = QVBoxLayout(self.detection_types_card)
        detection_types_layout.setContentsMargins(14, 14, 14, 14)
        detection_types_layout.setSpacing(8)
        detection_types_layout.addWidget(self.create_section_header(self.tr("检测方式"), "🔍"))
        detection_types_flow = FlowLayout(needAni=True)
        detection_types_flow.setContentsMargins(0, 0, 0, 0)
        detection_types_flow.setHorizontalSpacing(8)
        detection_types_flow.setVerticalSpacing(8)
        self.detection_mode_cards = {}
        detection_modes = (
            ("semantic_detect", self.tr("语意识别"), self.tr("按文本描述识别"), "🧠", "#EDE9FE"),
            ("space_detect", self.tr("框选区域"), self.tr("手动框选水印"), "🎯", "#DBEAFE"),
            ("subtitle", self.tr("字幕"), self.tr("识别字幕"), "💬", "#FCE7F3"),
            ("text_watermark", self.tr("文字水印"), self.tr("识别文字水印"), "🔤", "#DCFCE7"),
            ("general_watermark", self.tr("通用检测"), self.tr("检测 Logo 等内容"), "✨", "#FEF3C7"),
        )
        for mode, title, description, icon, icon_color in detection_modes:
            mode_card = OptionCard(
                title,
                description,
                icon=icon,
                icon_color=icon_color,
                recommended=mode in ("semantic_detect", "space_detect"),
            )
            mode_card.setMinimumWidth(95)
            mode_card.clicked.connect(lambda mode=mode: self._select_detection_mode(mode))
            detection_types_flow.addWidget(mode_card)
            self.detection_mode_cards[mode] = mode_card
        detection_types_layout.addLayout(detection_types_flow)
        ai_layout.addWidget(self.detection_types_card)

        self.parameters_card = QFrame()
        self.parameters_card.setObjectName("sectionCard")
        self.parameters_card.setProperty("parameterSection", True)
        parameters_layout = QVBoxLayout(self.parameters_card)
        parameters_layout.setContentsMargins(14, 14, 14, 14)
        parameters_layout.setSpacing(6)
        parameters_layout.addWidget(self.create_section_header(self.tr("检测参数"), "⚙️"))

        self.video_format_container = QWidget()
        video_format_layout = QVBoxLayout(self.video_format_container)
        video_format_layout.setContentsMargins(0, 0, 0, 0)
        video_format_layout.setSpacing(6)
        video_format_layout.addWidget(self.create_label_group(self.tr("水印形式"), desc=None))
        format_layout = QHBoxLayout()
        format_layout.setContentsMargins(0, 0, 0, 0)
        self.static_format_card = OptionCard(self.tr("固定位置水印"), self.tr("水印位置及外观不变"))
        self.static_format_card.selected = True
        self.static_format_card.update_style()
        self.static_format_card.clicked.connect(lambda: self._select_watermark_format("static_watermark", False))
        self.dynamic_format_card = OptionCard(self.tr("动态变化水印"), self.tr("水印位置或外观变化"))
        self.dynamic_format_card.clicked.connect(lambda: self._select_watermark_format("dynamic_watermark", True))
        format_layout.addWidget(self.static_format_card)
        format_layout.addWidget(self.dynamic_format_card)
        video_format_layout.addLayout(format_layout)

        self.video_strategy_container = QWidget()
        video_strategy_layout = QVBoxLayout(self.video_strategy_container)
        video_strategy_layout.setContentsMargins(0, 0, 0, 0)
        video_strategy_layout.setSpacing(6)
        video_strategy_layout.addWidget(self.create_label_group(self.tr("动态水印检测方式"), None))
        strategy_layout = QHBoxLayout()
        strategy_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_by_frame_card = OptionCard(self.tr("逐帧检测"), self.tr("逐帧识别水印"))
        self.frame_by_frame_card.clicked.connect(lambda: self._select_video_detection_strategy("frame_by_frame"))
        self.tracking_detection_card = OptionCard(self.tr("跟踪检测"), self.tr("跟踪关键帧水印"), icon="🎯", icon_color="#DBEAFE", recommended=True)
        self.tracking_detection_card.clicked.connect(lambda: self._select_video_detection_strategy("tracking"))
        strategy_layout.addWidget(self.tracking_detection_card, 1)
        strategy_layout.addWidget(self.frame_by_frame_card, 1)
        video_strategy_layout.addLayout(strategy_layout)

        self.anchor_frame_container, self.anchor_frame_spin = (self._create_anchor_frame_control())
        self.frame_range_container, self.frame_range_start, self.frame_range_end = (self._create_frame_range_controls())

        self.tracking_container = QWidget()
        tracking_layout = QVBoxLayout(self.tracking_container)
        tracking_layout.setContentsMargins(0, 0, 0, 0)
        self.tracking_btn = QPushButton(self.tr("🎯 启用跟踪(可选)"))
        self.tracking_btn.setObjectName("secondaryBtn")
        self.tracking_btn.setCursor(Qt.PointingHandCursor)
        self.tracking_btn.clicked.connect(self._open_video_object_detector)
        tracking_layout.addWidget(self.tracking_btn)

        self.video_options_card = QFrame()
        self.video_options_card.setObjectName("sectionCard")
        video_options_layout = QVBoxLayout(self.video_options_card)
        video_options_layout.setContentsMargins(14, 14, 14, 14)
        video_options_layout.setSpacing(8)
        video_options_layout.addWidget(self.create_section_header(self.tr("视频检测设置"), "🎞️"))
        video_options_layout.addWidget(self.video_format_container)
        video_options_layout.addWidget(self.anchor_frame_container)
        video_options_layout.addWidget(self.video_strategy_container)
        video_options_layout.addWidget(self.tracking_container)
        video_options_layout.addWidget(self.frame_range_container)
        self.video_options_card.setVisible(False)
        ai_layout.insertWidget(0, self.video_options_card)

        self.mode_parameters_stack = AutoSizeStackedWidget()
        self.mode_parameters = {}
        for mode, _, _, _, _ in detection_modes:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(4)
            if mode == "semantic_detect":
                page_layout.addWidget(self.create_label_group(self.tr("描述要识别的水印"), self.tr("英文效果更佳")))
                self.input_box = QLineEdit()
                self.input_box.setObjectName("inputBox")
                setFont(self.input_box, 12)
                self.input_box.setPlaceholderText(self.tr("文本描述（建议英文，可点击下方快捷按钮）..."))
                self.input_box.textChanged.connect(lambda text: self.watermarkDetectPrompt.emit(text))
                page_layout.addWidget(self.input_box)
                chip_layout = QHBoxLayout()
                chip_layout.setSpacing(12)
                text_map = (
                    (self.tr("水印"), "watermark"),
                    (self.tr("字幕"), "subtitle"),
                    (self.tr("水印和字幕"), "watermark,subtitle"),
                )
                for label, prompt in text_map:
                    chip = QPushButton(label)
                    chip.setObjectName("chip")
                    setFont(chip, 12)
                    chip.clicked.connect(lambda _, prompt=prompt: self.input_box.setText(prompt))
                    chip_layout.addWidget(chip)
                chip_layout.addStretch()
                page_layout.addLayout(chip_layout)
                page_layout.addWidget(self.create_label_group(self.tr("水印区域选择 (可选)"), self.tr("限定检测区域，不选择则检测整张图片")))
                self.semantic_area_btn = self._create_area_button()
                page_layout.addWidget(self.semantic_area_btn)
            elif mode == "space_detect":
                page_layout.addWidget(self.create_label_group(self.tr("精准框选"), self.tr("在图片区手动框选检测区域")))
                self.precise_select_btn = QPushButton(self.tr("🎯 开始框选"))
                self.precise_select_btn.setObjectName("secondaryBtn")
                self.precise_select_btn.clicked.connect(lambda _: self._watermark_area_selector())
                page_layout.addWidget(self.precise_select_btn)
            else:
                page_layout.addWidget(self.create_label_group(self.tr("水印区域选择 (可选)"), self.tr("限定检测区域，不选择则检测整张图片")))
                page_layout.addWidget(self._create_area_button())
            self.mode_parameters_stack.addWidget(page)
            self.mode_parameters[mode] = page
        self.confidence_container = self._create_confidence_widget()
        self.detection_settings_container = QWidget()
        detection_settings_layout = QVBoxLayout(self.detection_settings_container)
        detection_settings_layout.setContentsMargins(0, 0, 0, 0)
        detection_settings_layout.setSpacing(6)
        detection_settings_layout.addWidget(self.mode_parameters_stack)
        detection_settings_layout.addWidget(self.confidence_container)
        parameters_layout.addWidget(self.detection_settings_container)

        self.video_format_container.setVisible(False)
        self.video_strategy_container.setVisible(False)
        ai_layout.addWidget(self.parameters_card)

        manual_page = QWidget()
        manual_layout = QVBoxLayout(manual_page)
        manual_layout.setContentsMargins(0, 0, 0, 0)

        manual_layout.addWidget(self.create_label_group(self.tr("导入已有水印标注"), self.tr("若选择 Mask 目录，将跳过手动标注")))
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(8)
        self.dir_input = QLineEdit()
        self.dir_input.setObjectName("inputBox")
        self.dir_input.setReadOnly(True)
        self.dir_input.setPlaceholderText(self.tr("未选择目录（选填）..."))
        setFont(self.dir_input, 12)
        self.dir_select_btn = QPushButton(self.tr("打开"))
        self.dir_select_btn.setObjectName("secondaryBtn")
        self.dir_select_btn.setFixedHeight(40)
        setFont(self.dir_select_btn, 12, QFont.DemiBold)
        self.dir_select_btn.clicked.connect(self._on_select_mask_dir)
        self.dir_clear_btn = QPushButton("清除")
        self.dir_clear_btn.setObjectName("secondaryBtn")
        self.dir_clear_btn.setFixedHeight(40)
        self.dir_clear_btn.setVisible(False)
        self.dir_clear_btn.clicked.connect(self._on_clear_mask_dir)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(self.dir_clear_btn)
        dir_layout.addWidget(self.dir_select_btn)
        manual_layout.addLayout(dir_layout)
        manual_layout.addSpacing(6)

        icon = QLabel("🎨")
        icon.setStyleSheet("font-size: 36px;")
        icon.setAlignment(Qt.AlignCenter)
        t2 = QLabel(self.tr("画笔编辑模式"))
        setFont(t2, 14, QFont.Bold)
        t2.setObjectName("h2")
        t2.setAlignment(Qt.AlignCenter)
        d2 = QLabel(self.tr("请在工作区直接涂抹水印区域。"))
        d2.setObjectName("labelDesc")
        setFont(d2, 11)
        d2.setAlignment(Qt.AlignCenter)
        manual_layout.addWidget(icon)
        manual_layout.addWidget(t2)
        manual_layout.addWidget(d2)
        
        self.primary_btn = QPushButton(self.tr("开始标注水印"))
        self.primary_btn.setObjectName("primaryBtn")
        self.primary_btn.clicked.connect(lambda _: self._process_manual_detect())
        setFont(self.primary_btn, 13, QFont.DemiBold)
        manual_layout.addWidget(self.primary_btn)

        self.stack.addWidget(ai_page)
        self.stack.addWidget(manual_page)

        self.card_layout.addWidget(self.stack)
        self.main_layout.addWidget(self.card)

        self._select_detection_mode("semantic_detect", update_height=False)
        self._select_video_detection_strategy("tracking")

    def _create_area_button(self):
        button = QPushButton(self.tr("🖼 水印区域选择"))
        button.setObjectName("secondaryBtn")
        button.clicked.connect(lambda _: self._watermark_area_selector(image_boxes_only=True))
        return button

    def _create_confidence_widget(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        title = QLabel(self.tr("检测灵敏度"))
        title.setObjectName("labelTitle")
        setFont(title, 13, QFont.DemiBold)
        description = QLabel(self.tr("越高越严格，可能漏检；越低越宽松，可能误检"))
        description.setObjectName("labelDesc")
        setFont(description, 11)
        info_layout.addWidget(title)
        info_layout.addWidget(description)

        self.conf_val_label = QLabel("0.30")
        self.conf_val_label.setObjectName("confValBadge")
        setFont(self.conf_val_label, 12, QFont.Bold)
        header.addLayout(info_layout)
        header.addStretch()
        header.addWidget(self.conf_val_label)
        layout.addLayout(header)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setObjectName("confSlider")
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(30)
        self.conf_slider.setFixedHeight(24)
        self.conf_slider.setCursor(Qt.PointingHandCursor)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        layout.addWidget(self.conf_slider)
        return container

    def _create_anchor_frame_control(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.create_label_group(self.tr("起始关键帧"), None))

        frame_spin = QSpinBox()
        frame_spin.setObjectName("anchorFrameInput")
        frame_spin.setRange(1, self.total_video_frames)
        frame_spin.setValue(1)
        frame_spin.setFixedHeight(30)
        frame_layout = QHBoxLayout()
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(8)
        frame_layout.addWidget(frame_spin, 1)
        preview_button = QPushButton()
        preview_button.setObjectName("previewBtn")
        preview_button.setIcon(FluentIcon.VIDEO.icon())
        preview_button.setIconSize(QSize(16, 16))
        preview_button.setFixedSize(36, 30)
        preview_button.setToolTip(self.tr("预览视频"))
        preview_button.setAccessibleName(self.tr("预览视频"))
        preview_button.setCursor(Qt.PointingHandCursor)
        preview_button.clicked.connect(lambda _checked=False: self._open_video_preview(frame_spin.value()))
        frame_layout.addWidget(preview_button)
        layout.addLayout(frame_layout)
        return container, frame_spin

    def _create_frame_range_controls(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.create_label_group(self.tr("处理帧范围"), None))

        range_layout = QHBoxLayout()
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(8)
        range_start = QSpinBox()
        range_start.setObjectName("frameRangeInput")
        range_start.setRange(1, self.total_video_frames)
        range_start.setValue(1)
        range_start.setFixedHeight(30)
        range_end = QSpinBox()
        range_end.setObjectName("frameRangeInput")
        range_end.setRange(1, self.total_video_frames)
        range_end.setValue(self.total_video_frames)
        range_end.setFixedHeight(30)
        range_layout.addWidget(range_start, 1)
        range_label = QLabel(self.tr("至"))
        range_label.setObjectName("inlineLabel")
        range_layout.addWidget(range_label, 0, Qt.AlignCenter)
        range_layout.addWidget(range_end, 1)
        preview_button = QPushButton()
        preview_button.setObjectName("previewBtn")
        preview_button.setIcon(FluentIcon.VIDEO.icon())
        preview_button.setIconSize(QSize(16, 16))
        preview_button.setFixedSize(36, 30)
        preview_button.setToolTip(self.tr("预览视频"))
        preview_button.setAccessibleName(self.tr("预览视频"))
        preview_button.setCursor(Qt.PointingHandCursor)
        preview_button.clicked.connect(lambda _checked=False: self._open_video_preview(range_start.value()))
        range_layout.addWidget(preview_button)
        layout.addLayout(range_layout)
        return container, range_start, range_end

    def create_label_group(self, title, desc):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 2, 0, 2)
        l.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("labelTitle")
        setFont(t, 13, QFont.DemiBold)
        l.addWidget(t)
        if desc:
            d = QLabel(desc)
            d.setObjectName("labelDesc")
            setFont(d, 12)
            l.addWidget(d)
        return w

    def create_section_header(self, title, icon):
        header = QWidget()
        header.setStyleSheet("QWidget { background: transparent; }")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        icon_label = QLabel(icon)
        setFont(icon_label, 13)
        icon_label.setStyleSheet("color: #555555; background: transparent;")

        title_label = QLabel(title)
        setFont(title_label, 13, QFont.DemiBold)
        title_label.setStyleSheet("color: #555555; background: transparent;")

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color: #e2e8f0;")

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(divider, 1)
        return header
    
    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_initial_height_set", False):
            self._initial_height_set = True
            target_height = self._calc_card_target_height()
            self.card.setMaximumHeight(target_height)
            self.card.setMinimumHeight(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def _calc_visible_widget_height(self, widget: QWidget) -> int:
        if not widget.isVisible():
            return 0
        layout = widget.layout()
        if layout is None:
            return widget.sizeHint().height()
        m = layout.contentsMargins()
        v_margins = m.top() + m.bottom()
        h_margins = m.left() + m.right()

        if isinstance(widget, QStackedWidget):
            current = widget.currentWidget()
            if current is None:
                return 0
            return self._calc_visible_widget_height(current) + v_margins

        item_heights = []
        item_widths = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                if not w.isVisible():
                    continue
                item_heights.append(self._calc_visible_widget_height(w))
                item_widths.append(w.sizeHint().width() + h_margins)
            elif item.layout() is not None:
                sub_h = self._calc_visible_layout_height(item.layout())
                if sub_h > 0:
                    item_heights.append(sub_h)
            elif item.spacerItem() is not None:
                sp = item.spacerItem()
                item_heights.append(sp.sizeHint().height())

        if not item_heights:
            return v_margins

        if isinstance(layout, QHBoxLayout):
            total = max(item_heights) + v_margins
        else:
            spacing_total = layout.spacing() * max(0, len(item_heights) - 1)
            total = sum(item_heights) + spacing_total + v_margins
        return total

    def _calc_visible_layout_height(self, layout) -> int:
        m = layout.contentsMargins()
        v_margins = m.top() + m.bottom()
        item_heights = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                if not w.isVisible():
                    continue
                item_heights.append(self._calc_visible_widget_height(w))
            elif item.layout() is not None:
                h = self._calc_visible_layout_height(item.layout())
                if h > 0:
                    item_heights.append(h)
            elif item.spacerItem() is not None:
                item_heights.append(item.spacerItem().sizeHint().height())
        if not item_heights:
            return v_margins
        if isinstance(layout, QHBoxLayout):
            return max(item_heights) + v_margins
        spacing_total = layout.spacing() * max(0, len(item_heights) - 1)
        return sum(item_heights) + spacing_total + v_margins

    def _calc_card_target_height(self) -> int:
        tabs_h = self.tabs_container.sizeHint().height()
        current_page = self.stack.widget(self.current_tab_index)
        page_h = self._calc_visible_widget_height(current_page)
        extra = 8
        return tabs_h + extra + page_h

    def animate_height_change(self, target_height: int):
        start_height = self.card.height()
        self.height_anim = QPropertyAnimation(self.card, b"maximumHeight")
        self.height_anim.setDuration(250)
        self.height_anim.setStartValue(start_height)
        self.height_anim.setEndValue(target_height)
        self.height_anim.setEasingCurve(QEasingCurve.OutCubic)
        self.height_anim.finished.connect(lambda: self.card.setMinimumHeight(0))
        self.height_anim.start()

    def update_layout_height(self):
        target_height = self._calc_card_target_height()
        self.animate_height_change(target_height)

    def switch_tab(self, index):
        self.current_tab_index = index
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.tab_btns):
            btn.setChecked(i == index)
        if index == 0:
            self._select_detection_mode(self.current_detection_mode, update_height=False)
        else:
            self.watermarkDetectType.emit("manual_detect")
        self.update_layout_height()

    def _select_detection_mode(self, mode, update_height=True):
        self.current_detection_mode = mode
        for key, card in self.detection_mode_cards.items():
            card.selected = key == mode
            card.update_style()

        self.mode_parameters_stack.setCurrentWidget(self.mode_parameters[mode])
        self.mode_parameters_stack.updateGeometry()
        self.confidence_container.setVisible(mode in ("semantic_detect", "space_detect"))
        if mode in ("semantic_detect", "space_detect"):
            self.watermarkDetectType.emit("ai_interactive_detect")
            self.watermarkAIInteractiveType.emit(mode)
        else:
            self.watermarkDetectType.emit("ai_auto_detect")
            self.watermarkContent.emit(mode)

        self._update_tracking_visibility(update_height=False)
        if update_height:
            self.update_layout_height()

    def _select_watermark_format(self, watermark_format, is_dynamic):
        self.static_format_card.selected = not is_dynamic
        self.static_format_card.update_style()
        self.dynamic_format_card.selected = is_dynamic
        self.dynamic_format_card.update_style()
        self.watermarkFormat.emit(watermark_format)
        self.is_dynamic_watermark = is_dynamic
        self._update_tracking_visibility()

    def _select_video_detection_strategy(self, strategy):
        self.video_detection_strategy = strategy
        self.frame_by_frame_card.selected = strategy == "frame_by_frame"
        self.frame_by_frame_card.update_style()
        self.tracking_detection_card.selected = strategy == "tracking"
        self.tracking_detection_card.update_style()
        self._update_tracking_visibility()

    def apply_styles(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 8)
        self.card.setGraphicsEffect(shadow)

        self.setStyleSheet("""
            #mainWin { background-color: #F0F2F5; }
            #panelCard { background: white; border-radius: 24px; }
            #sectionCard { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; }
            #titleIndicator { background: #2563EB; border-radius: 2px; }
            #h2 { color: #0F172A; }
            
            #tabsContainer { background: #F1F5F9; border-radius: 14px; }
            #slider { background: white; border-radius: 10px; }
            #tabItem { 
                border: none; background: transparent; color: #64748B; padding: 12px; border-radius: 12px;
            }
            #tabItem:checked { color: #2563EB; background: white;}
            
            #labelTitle { color: #0F172A; }
            #labelDesc { color: #94A3B8; }
            QFrame[parameterSection="true"] QLabel#labelTitle {
                color: #0F172A;
                font-size: 13px;
            }

            QSpinBox#anchorFrameInput, QSpinBox#frameRangeInput {
                padding: 0 18px 0 10px;
                border-radius: 10px;
                color: #1e293b;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                font-size: 12px;
                selection-background-color: #6366f1;
            }
            QSpinBox#anchorFrameInput:hover, QSpinBox#frameRangeInput:hover {
                background: #fff;
                border-color: #cbd5e1;
            }
            QSpinBox#anchorFrameInput:focus, QSpinBox#frameRangeInput:focus {
                background: #fff;
                border-color: #818cf8;
            }
            QSpinBox#anchorFrameInput::up-button,
            QSpinBox#anchorFrameInput::down-button,
            QSpinBox#frameRangeInput::up-button,
            QSpinBox#frameRangeInput::down-button {
                subcontrol-origin: border;
                width: 13px;
                height: 12px;
                border: none;
                background: transparent;
            }
            QSpinBox#anchorFrameInput::up-button,
            QSpinBox#frameRangeInput::up-button {
                subcontrol-position: top right;
                top: 2px;
                right: 2px;
            }
            QSpinBox#anchorFrameInput::down-button,
            QSpinBox#frameRangeInput::down-button {
                subcontrol-position: bottom right;
                bottom: 2px;
                right: 2px;
            }
            QSpinBox#anchorFrameInput::up-arrow,
            QSpinBox#frameRangeInput::up-arrow {
                image: url(:/qfluentwidgets/images/icons/CareUpSolid_black.svg);
                width: 7px;
                height: 7px;
            }
            QSpinBox#anchorFrameInput::down-arrow,
            QSpinBox#frameRangeInput::down-arrow {
                image: url(:/qfluentwidgets/images/icons/CareDownSolid_black.svg);
                width: 7px;
                height: 7px;
            }
            
            OptionCard { background: #FAFBFC; border: 1.5px solid #E2E8F0; border-radius: 12px; }
            OptionCard:hover { border-color: #93C5FD; }
            OptionCard[selected="true"] { background: #EFF6FF; border-color: #2563EB; }
            #optTitle { color: #0F172A; }
            OptionCard[selected="true"] #optTitle { color: #2563EB; }
            #optDesc { color: #94A3B8; }
            #recommendBadge {
                color: #B45309;
                background-color: #FEF3C7;
                border-radius: 6px;
                padding: 2px 5px;
            }
            
            #inputBox { 
                padding: 8px; border: 1.5px solid #E2E8F0; border-radius: 12px; 
                background: white; color: #0F172A;
            }
            #inputBox:focus { border: 1.5px solid #2563EB; }
            
            #chip { 
                background: #F1F5F9; border: 1px solid #e2e8f0; border-radius: 8px; 
                padding: 5px 10px; color: #64748B; 
            }
            #chip:hover { background: #2563EB; color: white; }
            
            #secondaryBtn { 
                background: #F1F5F9; 
                border: 1px solid #E2E8F0; 
                border-radius: 12px; 
                padding: 12px; 
                font-weight: 600; 
                color: #0F172A;
            }
            #secondaryBtn:hover { 
                background: #E2E8F0; 
                border-color: #CBD5E1;
            }
            #secondaryBtn:pressed { 
                background: #CBD5E1; 
                padding-top: 14px;
                padding-bottom: 10px;
            }
            #previewBtn {
                background: #F1F5F9;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                padding: 0;
            }
            #previewBtn:hover {
                background: #E2E8F0;
                border-color: #CBD5E1;
            }
            #previewBtn:pressed {
                background: #CBD5E1;
            }
            #primaryBtn { 
                background: #2563EB; color: white; border: none; border-radius: 14px; 
                padding: 16px; margin-top: 10px;
            }
            #primaryBtn:hover { background: #1D4ED8; }
            #primaryBtn:disabled { background: #CBD5E1; color: #94A3B8; }
            #confValBadge {
                color: #2563EB;
                background-color: #EFF6FF;
                padding: 4px 10px;
                border-radius: 6px;
                min-width: 40px;
            }
            #confSlider::groove:horizontal {
                border-radius: 3px;
                height: 6px;
                background: #E2E8F0;
            }
            #confSlider::handle:horizontal {
                background: #2563EB;
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            #confSlider::sub-page:horizontal {
                background: #2563EB;
                border-radius: 3px;
            }
        """)

    def _on_select_mask_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("选择 Mask 目录"), 
            "", 
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0)
        )
        if dir_path:
            self.mask_dir_path = dir_path
            self.dir_input.setText(dir_path)
            self.dir_clear_btn.setVisible(True)
            self.dir_select_btn.setVisible(False)
            self.primary_btn.setEnabled(False)
            self.maskDirectoryChanged.emit(dir_path) if len(os.listdir(dir_path)) > 1 else self.maskDirectoryChanged.emit(os.path.join(dir_path, os.listdir(dir_path)[0]))

    def _on_clear_mask_dir(self):
        self.mask_dir_path = ""
        self.dir_input.clear()
        self.dir_clear_btn.setVisible(False)
        self.dir_select_btn.setVisible(True)
        self.primary_btn.setEnabled(True)
        self.maskDirectoryChanged.emit("")

    def _on_conf_changed(self, value):
        float_val = value / 100.0
        self.conf_val_label.setText(f"{float_val:.2f}")
        self.watermarkConfidence.emit(float_val)

    def set_file_path(self, file_path):
        if not file_path:
            self.file_path = None
            self.is_video_file = False
            self._update_tracking_visibility()
            return
        if os.path.isfile(file_path):
            self.file_path = file_path
        else:
            self.file_path = os.path.join(file_path, os.listdir(file_path)[0])
        self.is_video_file = get_file_type(self.file_path) == "video"
        if self.is_video_file:
            self._update_video_frame_count()
        self._update_tracking_visibility()

    def _update_video_frame_count(self):
        capture = open_video_capture(self.file_path)
        if not capture.isOpened():
            capture.release()
            self.total_video_frames = 1
            MessageBox(
                title=self.tr("无法读取视频"),
                content=self.tr("无法获取视频帧数，请检查文件格式或文件是否损坏。"),
                parent=self.window(),
            ).exec()
        else:
            reported_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            capture.release()
            self.total_video_frames = (
                max(1, int(reported_frames)) if reported_frames > 0 else 1
            )

        self.anchor_frame_spin.setRange(1, self.total_video_frames)
        self.anchor_frame_spin.setValue(1)
        self.frame_range_start.setRange(1, self.total_video_frames)
        self.frame_range_start.setValue(1)
        self.frame_range_end.setRange(1, self.total_video_frames)
        self.frame_range_end.setValue(self.total_video_frames)

    def _watermark_area_selector(self, single_area_only=False, image_boxes_only=False):
        if not self.file_path:
            return
        area_selector = AreaSelectorDialog(file_path=self.file_path, single_area_only=single_area_only, parent=self)
        area_selector.exec()
        boxes = []
        for item in area_selector.get_results():
            boxes.append((item["x"], item["y"], item["x"] + item["w"], item["y"] + item["h"]))
        if not image_boxes_only:
            self.watermarkBoxes.emit(boxes)
        else:
            self.imageBoxes.emit(boxes)

    def _process_manual_detect(self):
        if not self.file_path:
            return
        is_video = True if get_file_type(self.file_path)=="video" else False
        watermarkMaskTool = WatermarkMaskTool(file_path=self.file_path, is_video=is_video, parent=self.window())
        watermarkMaskTool.exec()
        mask_path = watermarkMaskTool.get_mask_path()
        self.manualWatermarktMaskPath.emit(mask_path)

    def _update_tracking_visibility(self, update_height=True):
        is_fixed = self.is_video_file and not self.is_dynamic_watermark
        is_dynamic = self.is_video_file and self.is_dynamic_watermark
        is_frame_by_frame = is_dynamic and self.video_detection_strategy == "frame_by_frame"
        is_tracking = is_dynamic and self.video_detection_strategy == "tracking"

        self.video_options_card.setVisible(self.is_video_file)
        self.video_format_container.setVisible(self.is_video_file)
        self.video_strategy_container.setVisible(is_dynamic)
        self.anchor_frame_container.setVisible(is_fixed)
        self.frame_range_container.setVisible(is_fixed or is_frame_by_frame)
        self.detection_settings_container.setVisible(not self.is_video_file or is_fixed or is_frame_by_frame)
        self.tracking_container.setVisible(is_tracking)
        self.detection_types_card.setVisible(not is_tracking)
        self.parameters_card.setVisible(not is_tracking)
        if update_height:
            self.update_layout_height()

    def _open_video_object_detector(self):
        if not self.file_path:
            return
        if not cfg.get(cfg.localWatermarkRemoveEnabled):
            MessageBox(
                title=self.tr("提醒"),
                content=self.tr("请先在设置页面打开 '水印移除' 能力开关并下载视频模型"),
                parent=self.window()
            ).exec()
            return
        dialog = VideoObjectDetectorWidget(file_path=self.file_path, parent=self.window())
        dialog.exec()

    def _open_video_preview(self, initial_frame):
        if not self.file_path:
            return
        dialog = WatermarkVideoPreviewDialog(
            file_path=self.file_path,
            total_frames=self.total_video_frames,
            initial_frame=initial_frame,
            parent=self.window(),
        )
        if not dialog.capture.isOpened():
            dialog.capture.release()
            MessageBox(
                title=self.tr("无法读取视频"),
                content=self.tr("无法打开视频预览，请检查文件格式或文件是否损坏。"),
                parent=self.window(),
            ).exec()
            return
        dialog.exec()

    def _on_tracking_data_ready(self, data):
        self.watermarkTrackingData.emit(data)