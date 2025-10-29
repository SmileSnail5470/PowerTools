from PySide6.QtCore import Qt, QUrl, QPropertyAnimation, QEasingCurve, QEvent, Property, QParallelAnimationGroup
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QSlider, QCheckBox, QStyle, QFrame, QGraphicsDropShadowEffect,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from app.ui.library.qfluentwidgets import setFont

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setVisible(False)
        self.setStyleSheet("""
            background: rgba(0,0,0,0.65);
            color: white;
            border-radius: 8px;
        """)
        label = QLabel(self.tr("加载中..."), self)
        setFont(label, 14)
        label.setAlignment(Qt.AlignCenter)
        lay = QVBoxLayout(self)
        lay.addStretch()
        lay.addWidget(label, alignment=Qt.AlignHCenter)
        lay.addStretch()

    def showEvent(self, e):
        self.resize(self.parent().size())
        super().showEvent(e)

    def resizeEvent(self, e):
        self.resize(self.parent().size())
        super().resizeEvent(e)


class VideoCard(QFrame):
    def __init__(self, label_text: str, media_url: str, parent=None):
        super().__init__(parent)
        self.setObjectName("videoCard")
        self.setStyleSheet("""
            QFrame#videoCard {
                background: white;
                border-radius: 8px;
                border: 0px;
            }
        """)
        self.video_widget = QVideoWidget(self)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        self.player.setSource(QUrl(media_url))

        # Label badge (top-left)
        badge = QLabel(label_text, self)
        badge.setStyleSheet("""
            background: rgba(32,31,30,0.9);
            color: white;
            padding: 6px 10px;
            border-radius: 4px;
            font-weight: 500;
        """)
        badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.overlay = LoadingOverlay(self)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(self.video_widget)
        badge.move(12, 12)
        badge.setFixedHeight(26)
        badge.adjustSize()

    def show_loading(self, visible: bool):
        self.overlay.setVisible(visible)

class FluentShadowButton(QWidget):
    def __init__(self, text="", icon=None, secondary=False, icon_only=False, parent=None):
        super().__init__(parent)
        self._offsetY = 4.0
        self._shadowOpacity = 30

        # 基础按钮
        self.button = QPushButton(text, self)
        self.button.setObjectName("fluentButton")
        self.button.setProperty("secondary", secondary)
        self.button.setProperty("iconOnly", icon_only)

        if icon:
            self.button.setIcon(icon)
            self.button.setIconSize(self.button.iconSize() * 1.2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)

        # 阴影效果
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setOffset(0, self._offsetY)
        self.shadow.setColor(QColor(0, 0, 0, self._shadowOpacity))
        self.button.setGraphicsEffect(self.shadow)

        # 动画组（偏移 + 透明度）
        self.anim_offset = QPropertyAnimation(self, b"offsetY", self)
        self.anim_offset.setDuration(200)
        self.anim_offset.setEasingCurve(QEasingCurve.OutCubic)

        self.anim_opacity = QPropertyAnimation(self, b"shadowOpacity", self)
        self.anim_opacity.setDuration(200)
        self.anim_opacity.setEasingCurve(QEasingCurve.OutCubic)

        self.anim_group = QParallelAnimationGroup(self)
        self.anim_group.addAnimation(self.anim_offset)
        self.anim_group.addAnimation(self.anim_opacity)

        self.button.installEventFilter(self)

        # 样式（优化后）
        self.setStyleSheet("""
            QPushButton#fluentButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0078d4, stop:1 #106ebe);
                color: white;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                border: none;
                min-height: 16px;
            }
            QPushButton#fluentButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #106ebe, stop:1 #0e5a9e);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            QPushButton#fluentButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0e5a9e, stop:1 #0a4a8a);
            }
            QPushButton#fluentButton[secondary="true"] {
                background: rgba(243,242,241,0.9);
                color: #323130;
                border: 1px solid rgba(210,208,206,0.6);
            }
            QPushButton#fluentButton[secondary="true"]:hover {
                background: rgba(243,242,241,1.0);
                border: 1px solid rgba(210,208,206,0.9);
            }
            QPushButton#fluentButton[iconOnly="true"] {
                padding: 8px;
                min-width: 24px;
                min-height: 24px;
            }
            QPushButton#fluentButton:disabled {
                background: rgba(200,200,200,0.6);
                color: rgba(100,100,100,0.6);
                border: 1px solid rgba(200,200,200,0.4);
            }
            """
        )

    def getOffsetY(self):
        return self._offsetY

    def setOffsetY(self, y):
        self._offsetY = y
        self.shadow.setOffset(0, y)
        self.button.graphicsEffect().update()
        self.button.update()

    offsetY = Property(float, getOffsetY, setOffsetY)

    def getShadowOpacity(self):
        return self._shadowOpacity

    def setShadowOpacity(self, alpha):
        self._shadowOpacity = alpha
        color = self.shadow.color()
        color.setAlpha(int(alpha))
        self.shadow.setColor(color)
        self.button.graphicsEffect().update()
        self.button.update()

    shadowOpacity = Property(float, getShadowOpacity, setShadowOpacity)

    # ---------- 悬浮动画控制 ----------
    def eventFilter(self, obj, event):
        if obj is self.button:
            if event.type() == QEvent.Enter:
                self._animate_shadow(target_y=7, target_alpha=60)
            elif event.type() == QEvent.Leave:
                self._animate_shadow(target_y=4, target_alpha=20)
        return super().eventFilter(obj, event)

    def _animate_shadow(self, target_y, target_alpha):
        self.anim_group.stop()
        self.anim_offset.setStartValue(self._offsetY)
        self.anim_offset.setEndValue(target_y)
        self.anim_opacity.setStartValue(self._shadowOpacity)
        self.anim_opacity.setEndValue(target_alpha)
        self.anim_group.start()

    def setIcon(self, icon):
        self.button.setIcon(icon)

    def setText(self, text):
        self.button.setText(text)

    def setToolTip(self, text):
        self.button.setToolTip(text)

    def setFixedSize(self, *args):
        self.button.setFixedSize(*args)

    def setMinimumHeight(self, h):
        self.button.setMinimumHeight(h)

    @property
    def clicked(self):
        return self.button.clicked

class SyncVideoViewer(QWidget):
    def __init__(self, video1: str = "", video2: str = "", parent=None):
        super().__init__(parent)
        if not video1:
            video1 = "https://www.w3schools.com/html/mov_bbb.mp4"
        if not video2:
            video2 = "https://www.w3schools.com/html/mov_bbb.mp4"
        self.video1_card = VideoCard(self.tr("视频 1 - 原始版本"), video1, self)
        self.video2_card = VideoCard(self.tr("视频 2 - 对比版本"), video2, self)

        style = self.style()
        self.play_pause_btn = FluentShadowButton(
            text=self.tr("播放"), 
            icon=style.standardIcon(QStyle.SP_MediaPlay)
        )
        self.stop_btn = FluentShadowButton(
            icon=style.standardIcon(QStyle.SP_MediaStop), 
            secondary=True, 
            icon_only=True
    )
        self.rewind_btn = FluentShadowButton(
            icon=style.standardIcon(QStyle.SP_MediaSeekBackward),
            secondary=True, 
            icon_only=True
        )
        self.forward_btn = FluentShadowButton(
            icon=style.standardIcon(QStyle.SP_MediaSeekForward), 
            secondary=True, 
            icon_only=True
        )

        self.stop_btn.setToolTip(self.tr("停止"))
        self.rewind_btn.setToolTip(self.tr("回退 10s"))
        self.forward_btn.setToolTip(self.tr("快进 10s"))

        # progress
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.current_time_label = QLabel("00:00")
        self.duration_label = QLabel("00:00")

        # volume and loop
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_value_label = QLabel("50%")
        self.loop_checkbox = QCheckBox(self.tr("循环播放"))
        self.sync_enabled = True

        # layout composition
        main_l = QVBoxLayout(self)
        main_l.setContentsMargins(20, 20, 20, 20)
        main_l.setSpacing(6)

        # video section
        videos_l = QVBoxLayout()
        videos_l.setSpacing(3)
        videos_l.addWidget(self.video1_card)
        videos_l.addWidget(self.video2_card)
        main_l.addLayout(videos_l, 9)

        # controls section (card-like)
        controls_frame = QFrame(self)
        controls_frame.setStyleSheet("""
            background: white;
            border-radius: 8px;
        """)
        controls_l = QVBoxLayout(controls_frame)
        controls_l.setContentsMargins(16, 12, 16, 12)
        controls_l.setSpacing(10)

        # header row (title + sync badge)
        header_l = QHBoxLayout()
        title = QLabel(self.tr("播放控制"))
        title.setStyleSheet("font-size:18px; font-weight:600;")
        header_l.addWidget(title)
        header_l.addStretch()
        sync_badge = QLabel(self.tr("●  同步已启用"))
        sync_badge.setStyleSheet("color:#0078D4; font-weight:500;")
        header_l.addWidget(sync_badge)
        controls_l.addLayout(header_l)

        # playback controls row
        pb_l = QHBoxLayout()
        pb_l.setSpacing(8)
        pb_l.addWidget(self.play_pause_btn)
        pb_l.addWidget(self.stop_btn)
        pb_l.addWidget(self.rewind_btn)
        pb_l.addWidget(self.forward_btn)
        pb_l.addStretch()
        controls_l.addLayout(pb_l)

        # progress section
        prog_l = QVBoxLayout()
        time_l = QHBoxLayout()
        time_l.addWidget(self.current_time_label)
        time_l.addStretch()
        time_l.addWidget(self.duration_label)
        prog_l.addLayout(time_l)
        prog_l.addWidget(self.progress_slider)
        controls_l.addLayout(prog_l)

        # controls grid: volume + loop
        grid_l = QHBoxLayout()
        vol_group = QHBoxLayout()
        vol_group.addWidget(QLabel("🔊 音量"))
        vol_group.addWidget(self.volume_slider)
        vol_group.addWidget(self.volume_value_label)
        grid_l.addLayout(vol_group)
        grid_l.addStretch()
        loop_group = QHBoxLayout()
        loop_group.addWidget(QLabel("🔄"))
        loop_group.addWidget(self.loop_checkbox)
        grid_l.addLayout(loop_group)
        controls_l.addLayout(grid_l)

        main_l.addWidget(controls_frame, 1)

        self.setStyleSheet("""
        QWidget {
            color: #323130;
            background: #F3F2F1;
        }

        /* ------------------- QPushButton ------------------- */
        QPushButton {
            background-color: #0078D4;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: 500;
            min-height: 32px;
        }
        QPushButton:hover {
            background-color: #106EBE;
        }
        QPushButton:pressed {
            background-color: #005A9E;
        }
        QPushButton:disabled {
            opacity: 0.5;
        }

        /* secondary 按钮 */
        QPushButton[secondary="true"] {
            background: #F3F2F1;
            color: #323130;
            border: 1px solid #D2D0CE;
        }
        QPushButton[secondary="true"]:hover {
            background: #EDEBE9;
            border-color: #C8C6C4;
        }

        /* iconOnly 小按钮 */
        QPushButton[iconOnly="true"] {
            padding: 4px;
            width: 32px;
            height: 32px;
            min-width: 32px;
            min-height: 32px;
        }
        QSlider::groove:horizontal {
            height: 4px;
            background: #E1E1E1;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            width: 12px;
            height: 12px;
            margin-top: -4px;
            margin-bottom: -4px;
            background: #0078D4;
            border: 2px solid white;
            border-radius: 6px;
        }
        QCheckBox::indicator {
            width: 40px;
            height: 20px;
            border-radius: 10px;
            background: #C8C6C4;
        }
        QCheckBox::indicator:unchecked, QCheckBox::indicator:checked {
            min-width: 16px;
            min-height: 16px;
            max-width: 16px;
            max-height: 16px;
            background: white;
            border-radius: 8px;
        }
        QCheckBox::indicator:unchecked {
            margin-left: 2px;
        }
        QCheckBox::indicator:checked {
            background: #0078D4; /* 背景颜色 */
            margin-left: 22px;
        }
        """)

        # state
        self.is_playing = False
        self.is_dragging = False
        self._internal_seek = False  # 用于避免 signal 互相触发

        # connect signals
        self._connect_signals()

        # show initial loading until media is ready
        self.video1_card.show_loading(True)
        self.video2_card.show_loading(True)

    def _connect_signals(self):
        v1p = self.video1_card.player
        v2p = self.video2_card.player

        # Play/pause/stop
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.stop_btn.clicked.connect(self.stop)
        self.rewind_btn.clicked.connect(lambda: self.seek_relative(-10_000))
        self.forward_btn.clicked.connect(lambda: self.seek_relative(10_000))

        # volume
        self.volume_slider.valueChanged.connect(self.on_volume_changed)

        # progress
        self.progress_slider.sliderPressed.connect(self.on_progress_pressed)
        self.progress_slider.sliderReleased.connect(self.on_progress_released)
        self.progress_slider.valueChanged.connect(self.on_progress_changed_by_ui)

        # loop checkbox
        self.loop_checkbox.stateChanged.connect(self.on_loop_changed)

        # players: position/duration/loading/status
        for player, card in ((v1p, self.video1_card), (v2p, self.video2_card)):
            player.positionChanged.connect(self._make_position_handler(player))
            player.durationChanged.connect(self._make_duration_handler(player))
            player.mediaStatusChanged.connect(self._make_status_handler(player, card))
            player.playbackStateChanged.connect(self._make_play_state_handler(player))

        # sync play/pause/seek between players (use flags to avoid recursion)
        v1p.playbackStateChanged.connect(lambda state: self._sync_playback_from(self.video1_card.player))
        v2p.playbackStateChanged.connect(lambda state: self._sync_playback_from(self.video2_card.player))

    def _make_position_handler(self, player):
        def handler(pos):
            # only update UI from primary source (use video1 as main)
            if player is self.video1_card.player and not self.is_dragging:
                dur = player.duration() or 0
                if dur > 0:
                    pct = pos / dur
                    self._internal_seek = True
                    self.progress_slider.setValue(int(pct * 1000))
                    self._internal_seek = False
                    self.current_time_label.setText(self._format_ms(pos))
            # if syncing enabled and not internal
            # other player's position set is handled in sync logic
        return handler

    def _make_duration_handler(self, player):
        def handler(dur):
            if player is self.video1_card.player:
                self.duration_label.setText(self._format_ms(dur))
        return handler

    def _make_status_handler(self, player, card):
        def handler(status):
            # mediaStatus: show loading until BufferedMedia or LoadedMedia
            from PySide6.QtMultimedia import QMediaPlayer
            if status in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia, QMediaPlayer.StalledMedia):
                card.show_loading(False)
            else:
                card.show_loading(True)
        return handler

    def _make_play_state_handler(self, player):
        def handler(state):
            # update play button text/icon when primary player's state changes
            if player is self.video1_card.player:
                if state == QMediaPlayer.PlayingState:
                    self.play_pause_btn.setText("暂停")
                    self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                    self.is_playing = True
                else:
                    self.play_pause_btn.setText("播放")
                    self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                    self.is_playing = False
        return handler

    def _sync_playback_from(self, source_player):
        """当某个 player 播放状态改变时，同步另一个 player（如果启用同步）"""
        if not self.sync_enabled:
            return
        target = self.video2_card.player if source_player is self.video1_card.player else self.video1_card.player
        # 同步状态
        s = source_player.playbackState()
        if s == QMediaPlayer.PlayingState:
            if target.playbackState() != QMediaPlayer.PlayingState:
                target.play()
        elif s == QMediaPlayer.PausedState:
            if target.playbackState() != QMediaPlayer.PausedState:
                target.pause()
        elif s == QMediaPlayer.StoppedState:
            target.stop()

        # 同步时间 (seek)
        # use setPosition carefully to avoid loops
        try:
            if not self.is_dragging:
                pos = source_player.position()
                if abs(pos - target.position()) > 300:  # 若差距 >300ms 才同步
                    self._internal_seek = True
                    target.setPosition(pos)
                    self._internal_seek = False
        except Exception:
            pass

    # ---------- controls ----------
    def toggle_play_pause(self):
        p1 = self.video1_card.player
        p2 = self.video2_card.player
        if p1.playbackState() == QMediaPlayer.PlayingState:
            p1.pause()
            p2.pause()
        else:
            p1.play()
            p2.play()

    def stop(self):
        self.video1_card.player.stop()
        self.video2_card.player.stop()

    def seek_relative(self, ms):
        # relative seek on both
        def clamp(x, lo, hi): return max(lo, min(hi, x))
        for p in (self.video1_card.player, self.video2_card.player):
            dur = p.duration() or 0
            new = clamp(p.position() + ms, 0, dur)
            p.setPosition(new)

    def on_volume_changed(self, value: int):
        vol = value / 100.0
        for card in (self.video1_card, self.video2_card):
            card.audio_output.setVolume(vol)
        self.volume_value_label.setText(f"{value}%")

    def on_loop_changed(self, state):
        loop = (state == Qt.Checked)
        # QMediaPlayer 没有直接 loop 属性，简单地在 mediaStatusChanged 检测到结束时重播
        # 这里用 timer 检测播放结束并重启，或用 positionChanged 与 duration 比较
        # 为简洁，在 playbackStateChanged 监听到停止时重启（若 loop true）
        if loop:
            # connect a small handler
            for p in (self.video1_card.player, self.video2_card.player):
                p.playbackStateChanged.connect(self._loop_handler)
        else:
            for p in (self.video1_card.player, self.video2_card.player):
                try:
                    p.playbackStateChanged.disconnect(self._loop_handler)
                except Exception:
                    pass

    def _loop_handler(self, state):
        # 如果停止或已到末尾，则重播
        if state == QMediaPlayer.StoppedState:
            if self.loop_checkbox.isChecked():
                self.video1_card.player.setPosition(0)
                self.video2_card.player.setPosition(0)
                self.video1_card.player.play()
                self.video2_card.player.play()

    def on_progress_pressed(self):
        self.is_dragging = True
        # pause both during drag
        self.video1_card.player.pause()
        self.video2_card.player.pause()

    def on_progress_released(self):
        self.is_dragging = False
        # apply slider position to both players
        pct = self.progress_slider.value() / 1000.0
        for p in (self.video1_card.player, self.video2_card.player):
            dur = p.duration() or 0
            p.setPosition(int(pct * dur))
        # resume if was playing
        if self.is_playing:
            self.video1_card.player.play()
            self.video2_card.player.play()

    def on_progress_changed_by_ui(self, val):
        if self._internal_seek:
            return
        if not self.is_dragging:
            return
        pct = val / 1000.0
        dur = self.video1_card.player.duration() or 0
        pos = int(pct * dur)
        # update time label immediately
        self.current_time_label.setText(self._format_ms(pos))
        # apply to both players while dragging (so preview)
        self.video1_card.player.setPosition(pos)
        self.video2_card.player.setPosition(pos)

    def _format_ms(self, ms: int) -> str:
        s = ms // 1000
        m = s // 60
        sec = s % 60
        return f"{m:02d}:{sec:02d}"