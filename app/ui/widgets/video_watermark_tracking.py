import cv2
from PySide6.QtCore import Qt, Signal, QEasingCurve
from PySide6.QtGui import QFont, QColor, QImage, QPixmap, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSlider,
    QLabel, QPushButton, QFrame, QGraphicsDropShadowEffect,
    QSizePolicy, QScrollArea, QWidget
)
from app.ui.library.qfluentwidgets import setFont, FlowLayout, SwitchButton, IndicatorPosition


class TimelineSliderWithMarkers(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._keyframes = set()
        self._end_frame = None
        self._total_frames = 1

    def set_markers(self, keyframes: set, end_frame, total_frames: int):
        self._keyframes = keyframes
        self._end_frame = end_frame
        self._total_frames = max(total_frames, 1)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._keyframes and self._end_frame is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        handle_width = 14
        half_handle = handle_width // 2
        track_left = half_handle
        track_right = self.width() - half_handle
        track_width = track_right - track_left
        track_center_y = self.height() // 2

        pen = QPen(QColor("#ffffff"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor("#ef4444")))

        for frame in self._keyframes:
            if self._total_frames > 1:
                ratio = (frame - 1) / (self._total_frames - 1)
                x = track_left + ratio * track_width
                painter.drawEllipse(int(x) - 3, track_center_y - 3, 6, 6)

        if self._end_frame is not None:
            painter.setBrush(QBrush(QColor("#10b981")))
            ratio = (self._end_frame - 1) / (self._total_frames - 1) if self._total_frames > 1 else 0
            x = track_left + ratio * track_width
            painter.drawEllipse(int(x) - 4, track_center_y - 4, 8, 8)

        painter.end()


class KeyframeTag(QFrame):
    removed = Signal(int)
    clicked = Signal(int)

    def __init__(self, frame: int, is_end_frame=False, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.is_end_frame = is_end_frame
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        frame_label = QLabel(f"第 {frame} 帧")
        setFont(frame_label, 11, QFont.Bold)
        layout.addWidget(frame_label, alignment=Qt.AlignVCenter)

        remove_btn = QPushButton("x")
        remove_btn.setFixedSize(16, 16)
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.setObjectName("tagRemoveBtn")
        setFont(remove_btn, 11)
        remove_btn.clicked.connect(lambda: self.removed.emit(self.frame))
        layout.addWidget(remove_btn, alignment=Qt.AlignVCenter)

        if is_end_frame:
            self.setObjectName("endFrameTag")
        else:
            self.setObjectName("keyframeTag")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.frame)


class VideoWatermarkTrackingDialog(QDialog):
    trackingDataReady = Signal(dict)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent=parent)
        self.file_path = file_path
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 1
        self.video_width = 0
        self.video_height = 0
        self.keyframes_set = set()
        self.end_frame = None
        self.tracking_enabled = True
        self.cap = None

        self.setWindowTitle(self.tr("视频水印自动跟踪设置"))
        self.setMinimumSize(800, 600)
        self.resize(1150, 850)

        self._init_video()
        self._init_ui()
        self._apply_styles()
        self._update_frame_display()

    def _init_video(self):
        self.cap = cv2.VideoCapture(self.file_path)
        if self.cap.isOpened():
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _init_ui(self):
        dialog_main_layout = QVBoxLayout(self)
        dialog_main_layout.setContentsMargins(0, 0, 0, 0)
        dialog_main_layout.setSpacing(0)
        dialog_main_layout.setAlignment(Qt.AlignTop)

        self.global_scroll_area = QScrollArea()
        self.global_scroll_area.setObjectName("globalScrollArea")
        self.global_scroll_area.setFrameShape(QFrame.NoFrame)
        self.global_scroll_area.setAlignment(Qt.AlignTop)
        self.global_scroll_area.setWidgetResizable(True)
        self.global_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.global_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.global_scroll_area.setViewportMargins(0, 0, 0, 0)
        dialog_main_layout.addWidget(self.global_scroll_area)

        scroll_central_widget = QWidget()
        scroll_central_widget.setObjectName("scrollCentralWidget")
        scroll_central_widget.setContentsMargins(0, 0, 0, 0)
        
        main_layout = QHBoxLayout(scroll_central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        left_panel = QFrame()
        left_panel.setObjectName("annotationContainer")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("panelHeader")
        header.setFixedHeight(40)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        title = QLabel(self.tr("视频水印自动跟踪设置"))
        setFont(title, 16, QFont.DemiBold)
        header_layout.addWidget(title)
        header_layout.addStretch()

        meta_layout = QHBoxLayout()
        meta_layout.setSpacing(14)

        resolution_label = QLabel(self.tr("分辨率:"))
        resolution_label.setObjectName("metaLabel")
        setFont(resolution_label, 13)
        meta_layout.addWidget(resolution_label)
        self.resolution_value = QLabel(f"{self.video_width} × {self.video_height}")
        setFont(self.resolution_value, 13, QFont.DemiBold)
        meta_layout.addWidget(self.resolution_value)

        divider1 = QFrame()
        divider1.setFixedSize(1, 12)
        divider1.setObjectName("metaDivider")
        meta_layout.addWidget(divider1)

        fps_label = QLabel("FPS:")
        fps_label.setObjectName("metaLabel")
        setFont(fps_label, 13)
        meta_layout.addWidget(fps_label)
        self.fps_value = QLabel(f"{self.fps:.0f}")
        setFont(self.fps_value, 13, QFont.DemiBold)
        meta_layout.addWidget(self.fps_value)

        divider2 = QFrame()
        divider2.setFixedSize(1, 12)
        divider2.setObjectName("metaDivider")
        meta_layout.addWidget(divider2)

        total_label = QLabel(self.tr("总帧数:"))
        total_label.setObjectName("metaLabel")
        setFont(total_label, 13)
        meta_layout.addWidget(total_label)
        self.total_frames_value = QLabel(f"{self.total_frames}")
        setFont(self.total_frames_value, 13, QFont.DemiBold)
        meta_layout.addWidget(self.total_frames_value)

        header_layout.addLayout(meta_layout)
        left_layout.addWidget(header)

        self.video_preview = QLabel()
        self.video_preview.setObjectName("videoPreview")
        self.video_preview.setAlignment(Qt.AlignCenter)
        self.video_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_preview, 1)

        control_panel = QFrame()
        control_panel.setObjectName("controlPanel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(24, 10, 24, 10)
        control_layout.setSpacing(8)

        self.timeline_slider = TimelineSliderWithMarkers(Qt.Horizontal)
        self.timeline_slider.setObjectName("timelineSlider")
        self.timeline_slider.setRange(1, max(self.total_frames, 1))
        self.timeline_slider.setValue(1)
        self.timeline_slider.setFixedHeight(24)
        self.timeline_slider.setCursor(Qt.PointingHandCursor)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        control_layout.addWidget(self.timeline_slider)

        action_bar = QHBoxLayout()
        action_bar.setSpacing(12)

        self.add_keyframe_btn = QPushButton(self.tr("+ 标记为关键帧"))
        self.add_keyframe_btn.setObjectName("btnPrimary")
        self.add_keyframe_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.add_keyframe_btn, 14, QFont.DemiBold)
        self.add_keyframe_btn.clicked.connect(self._add_keyframe)
        action_bar.addWidget(self.add_keyframe_btn)

        self.set_end_frame_btn = QPushButton(self.tr("→ 设为结束帧"))
        self.set_end_frame_btn.setObjectName("btnSuccess")
        self.set_end_frame_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.set_end_frame_btn, 14, QFont.DemiBold)
        self.set_end_frame_btn.clicked.connect(self._set_end_frame)
        action_bar.addWidget(self.set_end_frame_btn)

        self.prev_frame_btn = QPushButton("◀")
        self.prev_frame_btn.setObjectName("btnNav")
        self.prev_frame_btn.setFixedWidth(44)
        self.prev_frame_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.prev_frame_btn, 14)
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        action_bar.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("▶")
        self.next_frame_btn.setObjectName("btnNav")
        self.next_frame_btn.setFixedWidth(44)
        self.next_frame_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.next_frame_btn, 14)
        self.next_frame_btn.clicked.connect(self._next_frame)
        action_bar.addWidget(self.next_frame_btn)

        action_bar.addStretch()

        self.frame_counter = QLabel(f"第 1 帧 / {self.total_frames} 帧")
        self.frame_counter.setObjectName("frameCounter")
        setFont(self.frame_counter, 14, QFont.Medium)
        action_bar.addWidget(self.frame_counter)

        control_layout.addLayout(action_bar)
        left_layout.addWidget(control_panel)

        status_panel = QFrame()
        status_panel.setObjectName("statusPanel")
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(10, 10, 10, 10)
        status_layout.setSpacing(10)

        end_frame_section = QVBoxLayout()
        end_frame_section.setSpacing(8)
        end_frame_title = QLabel(self.tr("已指定的结束帧"))
        end_frame_title.setObjectName("statusTitle")
        setFont(end_frame_title, 12, QFont.DemiBold)
        end_frame_section.addWidget(end_frame_title)

        self.end_frame_container = QHBoxLayout()
        self.end_frame_container.setSpacing(8)
        self.end_frame_container.setContentsMargins(0, 0, 0, 0)
        self.end_frame_container.setAlignment(Qt.AlignVCenter)
        self.end_frame_placeholder = QLabel(self.tr("暂未设置结束帧"))
        self.end_frame_placeholder.setObjectName("noData")
        setFont(self.end_frame_placeholder, 12)
        self.end_frame_container.addWidget(self.end_frame_placeholder)
        self.end_frame_container.addStretch()
        end_frame_section.addLayout(self.end_frame_container)
        status_layout.addLayout(end_frame_section)

        keyframes_section = QVBoxLayout()
        keyframes_section.setSpacing(8)
        keyframes_title = QLabel(self.tr("已选择的关键帧列表"))
        keyframes_title.setObjectName("statusTitle")
        setFont(keyframes_title, 12, QFont.DemiBold)
        keyframes_section.addWidget(keyframes_title)

        self.keyframes_flow_layout = FlowLayout(None, needAni=False)
        self.keyframes_flow_layout.setAnimation(250, QEasingCurve.OutQuad)
        self.keyframes_flow_layout.setContentsMargins(10, 10, 10, 10)
        self.keyframes_flow_layout.setVerticalSpacing(8)
        self.keyframes_flow_layout.setHorizontalSpacing(4)

        self.keyframes_placeholder = QLabel(self.tr("暂无标记关键帧，请在上方选择并添加"))
        self.keyframes_placeholder.setObjectName("noData")
        setFont(self.keyframes_placeholder, 12)
        self.keyframes_flow_layout.addWidget(self.keyframes_placeholder)

        keyframes_section.addLayout(self.keyframes_flow_layout)
        status_layout.addLayout(keyframes_section)
        left_layout.addWidget(status_panel)
        
        bottom_action_bar = QHBoxLayout()
        bottom_action_bar.setContentsMargins(24, 6, 24, 6)

        self.preview_tracking_btn = QPushButton(self.tr("▶ 预览跟踪结果"))
        self.preview_tracking_btn.setObjectName("previewTrackingBtn")
        self.preview_tracking_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.preview_tracking_btn, 12, QFont.DemiBold)
        self.preview_tracking_btn.clicked.connect(self._on_preview_tracking)
        bottom_action_bar.addWidget(self.preview_tracking_btn)

        bottom_action_bar.addStretch()

        self.confirm_btn = QPushButton(self.tr("✓ 确认标注"))
        self.confirm_btn.setObjectName("confirmBtn")
        self.confirm_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.confirm_btn, 12, QFont.DemiBold)
        self.confirm_btn.clicked.connect(self._on_confirm)
        bottom_action_bar.addWidget(self.confirm_btn)

        self.tracking_switch_label = QLabel(self.tr("启用跟踪"))
        self.tracking_switch_label.setObjectName("trackingSwitchLabel")
        setFont(self.tracking_switch_label, 12)
        bottom_action_bar.addSpacing(16)
        bottom_action_bar.addWidget(self.tracking_switch_label)

        self.tracking_switch = SwitchButton(parent=self, indicatorPos=IndicatorPosition.RIGHT)
        self.tracking_switch.setChecked(True)
        self.tracking_switch.checkedChanged.connect(self._on_tracking_enabled_changed)
        bottom_action_bar.addWidget(self.tracking_switch)

        left_layout.addLayout(bottom_action_bar)
        main_layout.addWidget(left_panel, 1)

        right_panel = QFrame()
        right_panel.setObjectName("instructionCard")
        right_panel.setFixedWidth(280)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        instruction_title = QLabel(self.tr("ℹ️ 操作说明"))
        setFont(instruction_title, 15, QFont.DemiBold)
        instruction_title.setObjectName("rightPanelTitle")
        right_layout.addWidget(instruction_title)

        instructions = [
            self.tr('1. <b>画面预览：</b>直接通过鼠标拖动或点击时间轴进度条，即可高频滑动定位视频画面。'),
            self.tr('2. <b>微调画面：</b>点击 <code style="background:#f1f5f9;padding:2px 5px;border-radius:4px;color:#db2777;">◀</code> 或 <code style="background:#f1f5f9;padding:2px 5px;border-radius:4px;color:#db2777;">▶</code> 按钮进行精确到每一帧的微调。'),
            self.tr('3. <b>设置结束帧：</b>定位到终点位置后，点击绿色按钮，将该帧锁定为视频处理结束点。'),
            self.tr('4. <b>标记关键帧：</b>点击紫色按钮记录多段水印的关键标记帧。'),
            self.tr('5. <b>快捷跳转：</b>在底部已生成的标签上点击，视频将自动定位到对应帧。'),
        ]

        for text in instructions:
            item = QLabel(text)
            item.setTextFormat(Qt.RichText)
            item.setWordWrap(True)
            item.setObjectName("instructionItem")
            setFont(item, 12)
            right_layout.addWidget(item)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setObjectName("instructionSeparator")
        right_layout.addWidget(separator)

        concept_title = QLabel(self.tr("📖 概念说明"))
        setFont(concept_title, 15, QFont.DemiBold)
        concept_title.setObjectName("rightPanelTitle")
        right_layout.addWidget(concept_title)

        keyframe_explain = QLabel(self.tr(
            '<b>关键帧：</b>视频中水印发生大小、形状等变化的帧。'
            '标记多个关键帧后，系统会自动追踪水印在这些帧之间的位置变化，'
            '用于精确去除移动水印。'
        ))
        keyframe_explain.setTextFormat(Qt.RichText)
        keyframe_explain.setWordWrap(True)
        keyframe_explain.setObjectName("instructionItem")
        setFont(keyframe_explain, 12)
        right_layout.addWidget(keyframe_explain)

        endframe_explain = QLabel(self.tr(
            '<b>结束帧：</b>视频处理的终止点。'
            '设置结束帧后，系统只会处理从第一个关键帧到结束帧之间的内容，'
            '适用于只需去除视频前半段水印的场景。'
        ))
        endframe_explain.setTextFormat(Qt.RichText)
        endframe_explain.setWordWrap(True)
        endframe_explain.setObjectName("instructionItem")
        setFont(endframe_explain, 12)
        right_layout.addWidget(endframe_explain)

        right_layout.addStretch()
        main_layout.addWidget(right_panel)

        self.global_scroll_area.setWidget(scroll_central_widget)

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #f1f5f9;
            }
            #globalScrollArea {
                background: #f1f5f9;
                border: none;
            }
            #scrollCentralWidget {
                background: #f1f5f9;
            }
            #annotationContainer {
                background: #ffffff;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
            #panelHeader {
                background: #f8fafc;
                border-bottom: 1px solid #e2e8f0;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }
            #metaLabel {
                color: #64748b;
            }
            #panelHeader QLabel {
                color: #0f172a;
            }
            #metaDivider {
                background-color: #cbd5e1;
            }
            #videoPreview {
                background: #e2e8f0;
            }
            #controlPanel {
                background: #ffffff;
            }
            #timelineSlider::groove:horizontal {
                border-radius: 3px;
                height: 6px;
                background: #e2e8f0;
            }
            #timelineSlider::handle:horizontal {
                background: #4f46e5;
                border: 2px solid #ffffff;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            #timelineSlider::handle:horizontal:hover {
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            #timelineSlider::sub-page:horizontal {
                background: #e2e8f0;
                border-radius: 3px;
            }
            #timelineSlider::add-page:horizontal {
                background: #e2e8f0;
                border-radius: 3px;
            }
            #btnSuccess {
                background: #10b981;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
            }
            #btnSuccess:hover {
                background: #059669;
            }
            #rightPanelTitle {
                color: #0f172a;
                background: transparent;
            }
            #btnPrimary {
                background: #4f46e5;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
            }
            #btnPrimary:hover {
                background: #4338ca;
            }
            #btnNav {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px 12px;
                color: #0f172a;
            }
            #btnNav:hover {
                background: #f8fafc;
                border-color: #cbd5e1;
            }
            #frameCounter {
                color: #0f172a;
                background: #f1f5f9;
                padding: 6px 12px;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
            }
            #statusPanel {
                background: #f8fafc;
                border-top: 1px solid #e2e8f0;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }
            #statusTitle {
                color: #64748b;
            }
            #noData {
                color: #64748b;
            }
            #keyframeTag {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }
            #keyframeTag QLabel {
                color: #0f172a;
                background: transparent;
            }
            #keyframeTag #tagRemoveBtn {
                background: transparent;
                border: none;
                color: #9ca3af;
            }
            #keyframeTag #tagRemoveBtn:hover {
                color: #ef4444;
            }
            #endFrameTag {
                background: #ecfdf5;
                border: 1px solid #a7f3d0;
                border-radius: 6px;
            }
            #endFrameTag QLabel {
                color: #065f46;
                background: transparent;
            }
            #endFrameTag #tagRemoveBtn {
                background: transparent;
                border: none;
                color: #6ee7b7;
            }
            #endFrameTag #tagRemoveBtn:hover {
                color: #ef4444;
            }
            #confirmBtn {
                background: #4f46e5;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
            }
            #confirmBtn:hover {
                background: #4338ca;
            }
            #previewTrackingBtn {
                background: #f59e0b;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
            }
            #previewTrackingBtn:hover {
                background: #d97706;
            }
            #previewTrackingBtn:disabled {
                background: #cbd5e1;
                color: #94a3b8;
            }
            #trackingSwitchLabel {
                color: #0f172a;
            }
            #instructionCard {
                background: #ffffff;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
            #instructionItem {
                color: #64748b;
                line-height: 1.5;
            }
            #instructionSeparator {
                color: #e2e8f0;
                margin-top: 6px;
                margin-bottom: 6px;
            }
        """)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 25))
        shadow.setOffset(0, 4)
        self.findChild(QFrame, "annotationContainer").setGraphicsEffect(shadow)

        shadow2 = QGraphicsDropShadowEffect()
        shadow2.setBlurRadius(20)
        shadow2.setColor(QColor(0, 0, 0, 25))
        shadow2.setOffset(0, 4)
        self.findChild(QFrame, "instructionCard").setGraphicsEffect(shadow2)

    def _on_slider_changed(self, value):
        self.current_frame = value
        self._update_frame_display()

    def _prev_frame(self):
        if self.current_frame > 1:
            self.current_frame -= 1
            self.timeline_slider.setValue(self.current_frame)

    def _next_frame(self):
        if self.current_frame < self.total_frames:
            self.current_frame += 1
            self.timeline_slider.setValue(self.current_frame)

    def _set_end_frame(self):
        self.end_frame = self.current_frame
        self._render_status_tags()
        self._update_timeline_markers()

    def _add_keyframe(self):
        if self.current_frame not in self.keyframes_set:
            self.keyframes_set.add(self.current_frame)
            self._render_status_tags()
            self._update_timeline_markers()

    def _remove_keyframe(self, frame):
        self.keyframes_set.discard(frame)
        self._render_status_tags()
        self._update_timeline_markers()

    def _remove_end_frame(self, frame):
        self.end_frame = None
        self._render_status_tags()
        self._update_timeline_markers()

    def _jump_to_frame(self, frame):
        self.current_frame = frame
        self.timeline_slider.setValue(frame)

    def _update_timeline_markers(self):
        self.timeline_slider.set_markers(self.keyframes_set, self.end_frame, self.total_frames)

    def _update_frame_display(self):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

                # 使用当前预览标签的实际大小进行缩放
                preview_size = self.video_preview.size()
                if preview_size.width() > 0 and preview_size.height() > 0:
                    scaled_pixmap = pixmap.scaled(
                        preview_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.video_preview.setPixmap(scaled_pixmap)
                else:
                    self.video_preview.setPixmap(pixmap.scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.frame_counter.setText(f"第 {self.current_frame} 帧 / {self.total_frames} 帧")

    def _render_status_tags(self):
        while self.end_frame_container.count():
            item = self.end_frame_container.takeAt(0)
            if item is not None:
                widget = item.widget() if hasattr(item, 'widget') else item
                if widget is not None:
                    widget.deleteLater()
        if self.end_frame is not None:
            tag = KeyframeTag(self.end_frame, is_end_frame=True, parent=self)
            tag.removed.connect(self._remove_end_frame)
            tag.clicked.connect(self._jump_to_frame)
            self.end_frame_container.addWidget(tag)
        else:
            placeholder = QLabel(self.tr("暂未设置结束帧"))
            placeholder.setObjectName("noData")
            setFont(placeholder, 12)
            self.end_frame_container.addWidget(placeholder)
        self.end_frame_container.addStretch()

        while self.keyframes_flow_layout.count():
            item = self.keyframes_flow_layout.takeAt(0)
            if item is not None:
                widget = item.widget() if hasattr(item, 'widget') else item
                if widget is not None:
                    widget.deleteLater()
        if not self.keyframes_set:
            placeholder = QLabel(self.tr("暂无标记关键帧，请在上方选择并添加"))
            placeholder.setObjectName("noData")
            setFont(placeholder, 12)
            self.keyframes_flow_layout.addWidget(placeholder)
        else:
            sorted_frames = sorted(self.keyframes_set)
            for frame in sorted_frames:
                tag = KeyframeTag(frame, is_end_frame=False, parent=self)
                tag.removed.connect(self._remove_keyframe)
                tag.clicked.connect(self._jump_to_frame)
                self.keyframes_flow_layout.addWidget(tag)

    def _on_preview_tracking(self):
        """Preview tracking results by running the tracking algorithm and displaying result frames."""
        if not self.keyframes_set:
            return

        tracking_data = {
            "file_path": self.file_path,
            "keyframes": sorted(self.keyframes_set),
            "end_frame": self.end_frame,
            "total_frames": self.total_frames,
            "fps": self.fps,
        }

        self.preview_tracking_btn.setEnabled(False)
        self.preview_tracking_btn.setText(self.tr("⏳ 跟踪中..."))

        # Call tracking algorithm interface - returns list of (frame_index, numpy_image) tuples
        tracked_frames = self._run_tracking_algorithm(tracking_data)

        self.preview_tracking_btn.setEnabled(True)
        self.preview_tracking_btn.setText(self.tr("▶ 预览跟踪结果"))

        if tracked_frames:
            self._display_tracked_frame(tracked_frames[0])
            self._tracked_results = tracked_frames
            self._tracked_preview_index = 0

    def _run_tracking_algorithm(self, tracking_data: dict):
        """
        Interface for the tracking algorithm.
        
        Args:
            tracking_data: dict containing:
                - file_path: str, path to the video file
                - keyframes: list[int], sorted list of keyframe indices
                - end_frame: int or None, the end frame index
                - total_frames: int, total number of frames in the video
                - fps: float, frames per second
        
        Returns:
            list of tuples: [(frame_index, numpy_bgr_image), ...] 
            Each tuple contains the frame number and the tracked result image (BGR numpy array).
            Returns empty list if tracking fails or is not implemented.
        """
        # TODO: Integrate actual tracking algorithm here
        # Example implementation placeholder:
        # from app.algorithms.tracker import run_watermark_tracking
        # return run_watermark_tracking(tracking_data)
        return []

    def _display_tracked_frame(self, tracked_frame):
        """Display a single tracked result frame in the video preview area.
        
        Args:
            tracked_frame: tuple of (frame_index, numpy_bgr_image)
        """
        frame_index, frame_bgr = tracked_frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        preview_size = self.video_preview.size()
        if preview_size.width() > 0 and preview_size.height() > 0:
            scaled_pixmap = pixmap.scaled(
                preview_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_preview.setPixmap(scaled_pixmap)
        else:
            self.video_preview.setPixmap(pixmap.scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.frame_counter.setText(f"第 {frame_index} 帧 / {self.total_frames} 帧 (跟踪结果)")

    def _on_tracking_enabled_changed(self, is_checked: bool):
        """Handle tracking switch toggle."""
        self.tracking_enabled = is_checked

    def _on_confirm(self):
        data = {
            "keyframes": sorted(self.keyframes_set),
            "end_frame": self.end_frame,
            "tracking_enabled": self.tracking_enabled,
        }
        self.trackingDataReady.emit(data)
        self.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_frame_display()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
            self.cap = None
        super().closeEvent(event)

    def reject(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        super().reject()