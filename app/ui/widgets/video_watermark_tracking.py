import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QSlider,
    QLabel, QPushButton, QFrame, QScrollArea, QGraphicsDropShadowEffect,
    QSizePolicy
)
from app.ui.library.qfluentwidgets import setFont


class KeyframeTag(QFrame):
    removed = Signal(int)
    clicked = Signal(int)

    def __init__(self, frame: int, fps: float, is_end_frame=False, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.is_end_frame = is_end_frame
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        time_str = self._format_time(frame / fps)
        frame_label = QLabel(f"第 {frame} 帧")
        setFont(frame_label, 12, QFont.Bold)
        layout.addWidget(frame_label)

        time_label = QLabel(f"({time_str})")
        setFont(time_label, 12)
        time_label.setObjectName("tagTimeLabel")
        layout.addWidget(time_label)

        remove_btn = QPushButton("×")
        remove_btn.setFixedSize(20, 20)
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.setObjectName("tagRemoveBtn")
        remove_btn.clicked.connect(lambda: self.removed.emit(self.frame))
        layout.addWidget(remove_btn)

        if is_end_frame:
            self.setObjectName("endFrameTag")
        else:
            self.setObjectName("keyframeTag")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.frame)

    @staticmethod
    def _format_time(seconds):
        if seconds < 0:
            return "00:00:00"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


class VideoWatermarkTrackingDialog(QDialog):
    trackingDataReady = Signal(dict)  # 输出: {"keyframes": [...], "end_frame": int|None}

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent=parent)
        self.file_path = file_path
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 0
        self.video_width = 0
        self.video_height = 0
        self.keyframes_set = set()
        self.end_frame = None
        self.cap = None

        self.setWindowTitle(self.tr("视频水印自动跟踪设置"))
        self.setMinimumSize(1100, 750)
        self.resize(1100, 750)

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
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ===== 左侧主面板 =====
        left_panel = QFrame()
        left_panel.setObjectName("annotationContainer")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # --- 标题栏 ---
        header = QFrame()
        header.setObjectName("panelHeader")
        header.setFixedHeight(56)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        title = QLabel(self.tr("视频水印自动跟踪设置"))
        setFont(title, 16, QFont.DemiBold)
        header_layout.addWidget(title)
        header_layout.addStretch()

        # 视频元信息
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

        # --- 视频预览区 ---
        self.video_preview = QLabel()
        self.video_preview.setObjectName("videoPreview")
        self.video_preview.setAlignment(Qt.AlignCenter)
        self.video_preview.setMinimumHeight(300)
        self.video_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_preview, 1)

        # --- 控制面板 ---
        control_panel = QFrame()
        control_panel.setObjectName("controlPanel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(24, 20, 24, 20)
        control_layout.setSpacing(16)

        # 时间轴滑块
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setObjectName("timelineSlider")
        self.timeline_slider.setRange(0, max(self.total_frames - 1, 1))
        self.timeline_slider.setValue(0)
        self.timeline_slider.setFixedHeight(24)
        self.timeline_slider.setCursor(Qt.PointingHandCursor)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        control_layout.addWidget(self.timeline_slider)

        # 按钮操作栏
        action_bar = QHBoxLayout()
        action_bar.setSpacing(12)

        # 设为结束帧按钮
        self.set_end_frame_btn = QPushButton(self.tr("→ 设为结束帧"))
        self.set_end_frame_btn.setObjectName("btnSuccess")
        self.set_end_frame_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.set_end_frame_btn, 14, QFont.DemiBold)
        self.set_end_frame_btn.clicked.connect(self._set_end_frame)
        action_bar.addWidget(self.set_end_frame_btn)

        # 标记为关键帧按钮
        self.add_keyframe_btn = QPushButton(self.tr("+ 标记为关键帧"))
        self.add_keyframe_btn.setObjectName("btnPrimary")
        self.add_keyframe_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.add_keyframe_btn, 14, QFont.DemiBold)
        self.add_keyframe_btn.clicked.connect(self._add_keyframe)
        action_bar.addWidget(self.add_keyframe_btn)

        # 上一帧按钮
        self.prev_frame_btn = QPushButton("◀")
        self.prev_frame_btn.setObjectName("btnNav")
        self.prev_frame_btn.setFixedWidth(44)
        self.prev_frame_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.prev_frame_btn, 14)
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        action_bar.addWidget(self.prev_frame_btn)

        # 下一帧按钮
        self.next_frame_btn = QPushButton("▶")
        self.next_frame_btn.setObjectName("btnNav")
        self.next_frame_btn.setFixedWidth(44)
        self.next_frame_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.next_frame_btn, 14)
        self.next_frame_btn.clicked.connect(self._next_frame)
        action_bar.addWidget(self.next_frame_btn)

        action_bar.addStretch()

        # 帧计数器
        self.frame_counter = QLabel(f"第 0 帧 / {self.total_frames} 帧")
        self.frame_counter.setObjectName("frameCounter")
        setFont(self.frame_counter, 14, QFont.Medium)
        action_bar.addWidget(self.frame_counter)

        control_layout.addLayout(action_bar)
        left_layout.addWidget(control_panel)

        # --- 底部状态信息区 ---
        status_panel = QFrame()
        status_panel.setObjectName("statusPanel")
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(24, 18, 24, 18)
        status_layout.setSpacing(16)

        # 结束帧区域
        end_frame_section = QVBoxLayout()
        end_frame_section.setSpacing(8)
        end_frame_title = QLabel(self.tr("已指定的结束帧"))
        end_frame_title.setObjectName("statusTitle")
        setFont(end_frame_title, 13, QFont.DemiBold)
        end_frame_section.addWidget(end_frame_title)

        self.end_frame_container = QHBoxLayout()
        self.end_frame_container.setSpacing(8)
        self.end_frame_placeholder = QLabel(self.tr("暂未设置结束帧"))
        self.end_frame_placeholder.setObjectName("noData")
        setFont(self.end_frame_placeholder, 13)
        self.end_frame_container.addWidget(self.end_frame_placeholder)
        self.end_frame_container.addStretch()
        end_frame_section.addLayout(self.end_frame_container)
        status_layout.addLayout(end_frame_section)

        # 关键帧列表区域
        keyframes_section = QVBoxLayout()
        keyframes_section.setSpacing(8)
        keyframes_title = QLabel(self.tr("已选择的关键帧列表"))
        keyframes_title.setObjectName("statusTitle")
        setFont(keyframes_title, 13, QFont.DemiBold)
        keyframes_section.addWidget(keyframes_title)

        # 关键帧标签滚动区域
        self.keyframes_scroll = QScrollArea()
        self.keyframes_scroll.setWidgetResizable(True)
        self.keyframes_scroll.setFixedHeight(60)
        self.keyframes_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.keyframes_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.keyframes_scroll.setObjectName("keyframesScroll")

        self.keyframes_widget = QWidget()
        self.keyframes_flow_layout = QHBoxLayout(self.keyframes_widget)
        self.keyframes_flow_layout.setContentsMargins(0, 0, 0, 0)
        self.keyframes_flow_layout.setSpacing(8)
        self.keyframes_flow_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.keyframes_placeholder = QLabel(self.tr("暂无标记关键帧，请在上方选择并添加"))
        self.keyframes_placeholder.setObjectName("noData")
        setFont(self.keyframes_placeholder, 13)
        self.keyframes_flow_layout.addWidget(self.keyframes_placeholder)

        self.keyframes_scroll.setWidget(self.keyframes_widget)
        keyframes_section.addWidget(self.keyframes_scroll)
        status_layout.addLayout(keyframes_section)

        # 确认按钮
        confirm_layout = QHBoxLayout()
        confirm_layout.addStretch()
        self.confirm_btn = QPushButton(self.tr("✓ 确认标注"))
        self.confirm_btn.setObjectName("confirmBtn")
        self.confirm_btn.setCursor(Qt.PointingHandCursor)
        setFont(self.confirm_btn, 14, QFont.DemiBold)
        self.confirm_btn.clicked.connect(self._on_confirm)
        confirm_layout.addWidget(self.confirm_btn)
        status_layout.addLayout(confirm_layout)

        left_layout.addWidget(status_panel)

        main_layout.addWidget(left_panel, 1)

        # ===== 右侧操作说明面板 =====
        right_panel = QFrame()
        right_panel.setObjectName("instructionCard")
        right_panel.setFixedWidth(260)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(12)

        instruction_title = QLabel(self.tr("ℹ️ 操作说明"))
        setFont(instruction_title, 15, QFont.DemiBold)
        right_layout.addWidget(instruction_title)

        instructions = [
            self.tr("1. 画面预览：直接通过鼠标拖动或点击时间轴进度条，即可高频滑动定位视频画面。"),
            self.tr("2. 微调画面：点击 ◀ 或 ▶ 按钮进行精确到每一帧的微调。"),
            self.tr("3. 设置结束帧：定位到终点位置后，点击绿色按钮，将该帧锁定为视频处理结束点。"),
            self.tr("4. 标记关键帧：点击蓝色按钮记录多段水印的关键标记帧。"),
            self.tr("5. 快捷跳转：在底部已生成的标签上点击，视频将自动定位到对应帧。"),
        ]

        for text in instructions:
            item = QLabel(text)
            item.setWordWrap(True)
            item.setObjectName("instructionItem")
            setFont(item, 13)
            right_layout.addWidget(item)

        right_layout.addStretch()
        main_layout.addWidget(right_panel)

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #f1f5f9;
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
                color: #334155;
            }
            #panelHeader QLabel {
                color: #0f172a;
            }
            #metaDivider {
                background-color: #cbd5e1;
            }
            #videoPreview {
                background: #0f172a;
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
            #timelineSlider::sub-page:horizontal {
                background: #4f46e5;
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
                font-style: italic;
            }
            #keyframesScroll {
                background: #f8fafc;
                border: none;
            }
            #keyframesScroll QWidget {
                background: #f8fafc;
            }
            #keyframeTag {
                background: #ffffff;
                border: 1px solid #a7f3d0;
                border-radius: 6px;
                padding: 4px 10px;
            }
            #keyframeTag QLabel {
                color: #065f46;
                background: transparent;
            }
            #keyframeTag #tagTimeLabel {
                color: #047857;
            }
            #keyframeTag #tagRemoveBtn {
                background: transparent;
                border: none;
                color: #6ee7b7;
                font-size: 16px;
                font-weight: bold;
            }
            #keyframeTag #tagRemoveBtn:hover {
                color: #ef4444;
            }
            #endFrameTag {
                background: #ffffff;
                border: 1px solid #a7f3d0;
                border-radius: 6px;
                padding: 4px 10px;
            }
            #endFrameTag QLabel {
                color: #065f46;
                background: transparent;
            }
            #endFrameTag #tagTimeLabel {
                color: #047857;
            }
            #endFrameTag #tagRemoveBtn {
                background: transparent;
                border: none;
                color: #6ee7b7;
                font-size: 16px;
                font-weight: bold;
            }
            #endFrameTag #tagRemoveBtn:hover {
                color: #ef4444;
            }
            #confirmBtn {
                background: #4f46e5;
                color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 12px 28px;
            }
            #confirmBtn:hover {
                background: #4338ca;
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
        """)

        # 阴影效果
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
        """时间轴滑块值改变"""
        self.current_frame = value
        self._update_frame_display()

    def _prev_frame(self):
        """上一帧"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.timeline_slider.setValue(self.current_frame)

    def _next_frame(self):
        """下一帧"""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.timeline_slider.setValue(self.current_frame)

    def _set_end_frame(self):
        """设置当前帧为结束帧"""
        self.end_frame = self.current_frame
        self._render_status_tags()

    def _add_keyframe(self):
        """标记当前帧为关键帧"""
        if self.current_frame not in self.keyframes_set:
            self.keyframes_set.add(self.current_frame)
            self._render_status_tags()

    def _remove_keyframe(self, frame):
        """移除关键帧"""
        self.keyframes_set.discard(frame)
        self._render_status_tags()

    def _remove_end_frame(self, frame):
        """移除结束帧"""
        self.end_frame = None
        self._render_status_tags()

    def _jump_to_frame(self, frame):
        """跳转到指定帧"""
        self.current_frame = frame
        self.timeline_slider.setValue(frame)

    def _update_frame_display(self):
        """更新视频帧显示"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

                # 缩放以适应预览区域
                preview_size = self.video_preview.size()
                if preview_size.width() > 0 and preview_size.height() > 0:
                    scaled_pixmap = pixmap.scaled(
                        preview_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.video_preview.setPixmap(scaled_pixmap)
                else:
                    self.video_preview.setPixmap(pixmap.scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 更新帧计数器
        self.frame_counter.setText(f"第 {self.current_frame} 帧 / {self.total_frames} 帧")

    def _render_status_tags(self):
        """重新渲染底部状态区域的标签"""
        # 清空结束帧容器
        while self.end_frame_container.count():
            item = self.end_frame_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self.end_frame is not None:
            tag = KeyframeTag(self.end_frame, self.fps, is_end_frame=True, parent=self)
            tag.removed.connect(self._remove_end_frame)
            tag.clicked.connect(self._jump_to_frame)
            self.end_frame_container.addWidget(tag)
        else:
            placeholder = QLabel(self.tr("暂未设置结束帧"))
            placeholder.setObjectName("noData")
            setFont(placeholder, 13)
            self.end_frame_container.addWidget(placeholder)
        self.end_frame_container.addStretch()

        # 清空关键帧容器
        while self.keyframes_flow_layout.count():
            item = self.keyframes_flow_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.keyframes_set:
            placeholder = QLabel(self.tr("暂无标记关键帧，请在上方选择并添加"))
            placeholder.setObjectName("noData")
            setFont(placeholder, 13)
            self.keyframes_flow_layout.addWidget(placeholder)
        else:
            sorted_frames = sorted(self.keyframes_set)
            for frame in sorted_frames:
                tag = KeyframeTag(frame, self.fps, is_end_frame=False, parent=self)
                tag.removed.connect(self._remove_keyframe)
                tag.clicked.connect(self._jump_to_frame)
                self.keyframes_flow_layout.addWidget(tag)

    def _on_confirm(self):
        """确认标注并发射信号"""
        data = {
            "keyframes": sorted(self.keyframes_set),
            "end_frame": self.end_frame,
            "fps": self.fps,
            "total_frames": self.total_frames,
        }
        self.trackingDataReady.emit(data)
        self.accept()

    def get_tracking_data(self) -> dict:
        """获取标注数据"""
        return {
            "keyframes": sorted(self.keyframes_set),
            "end_frame": self.end_frame,
            "fps": self.fps,
            "total_frames": self.total_frames,
        }

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 刷新视频帧以适应新尺寸
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
