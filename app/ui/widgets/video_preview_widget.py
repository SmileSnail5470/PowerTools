from PySide6.QtCore import Qt, Signal, Slot, QUrl
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


class SyncVideoView(QWidget):
    positionChanged = Signal(int)
    durationChanged = Signal(int)
    stateChanged = Signal(QMediaPlayer.PlaybackState)

    def __init__(self, video_path: str = "", parent=None, sub_title: str = ""):
        super().__init__(parent)
        self.sub_title = sub_title
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        # 视频输出组件
        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

        # 控制区
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(40)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.slider)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_widget)
        layout.addLayout(control_layout)

        # 信号绑定
        self.play_button.clicked.connect(self.toggle_play)
        self.slider.sliderMoved.connect(self.set_position)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._on_state_changed)

        if video_path:
            self.set_video(video_path)

    def set_video(self, path: str):
        self.player.setSource(QUrl.fromLocalFile(path))

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_position_changed(self, position):
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)
        self.positionChanged.emit(position)

    def _on_duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.durationChanged.emit(duration)

    def _on_state_changed(self, state):
        self.play_button.setText("⏸" if state == QMediaPlayer.PlayingState else "▶")
        self.stateChanged.emit(state)

    @Slot(int)
    def set_position(self, position):
        self.player.setPosition(position)

    @Slot(int)
    def sync_position(self, position):
        """用于同步另一个视频位置"""
        if abs(self.player.position() - position) > 50:  # 允许 50ms 容差
            self.player.setPosition(position)

    @Slot(QMediaPlayer.PlaybackState)
    def sync_state(self, state):
        """同步播放/暂停状态"""
        if self.player.playbackState() != state:
            if state == QMediaPlayer.PlayingState:
                self.player.play()
            elif state == QMediaPlayer.PausedState:
                self.player.pause()


class SyncVideoViewer(QWidget):
    """双视频同步查看器"""
    def __init__(self, video1: str = "", video2: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.view1 = SyncVideoView(video1, sub_title="原视频预览区域")
        self.view2 = SyncVideoView(video2, sub_title="添加水印后预览区域")

        layout.addWidget(self.view1)
        layout.addWidget(self.view2)

        # 信号互联实现同步
        self.view1.positionChanged.connect(self.view2.sync_position)
        self.view2.positionChanged.connect(self.view1.sync_position)
        self.view1.stateChanged.connect(self.view2.sync_state)
        self.view2.stateChanged.connect(self.view1.sync_state)

        self.setStyleSheet("""
            QWidget {
                background-color: #f3f3f3;
                border-radius: 8px;
            }
            QVideoWidget {
                background-color: #000000;
                border: 1px solid #ccc;
                border-radius: 6px;
            }
            QPushButton {
                border: none;
                background-color: transparent;
                font-size: 16px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #ddd;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                background: #0078d7;
                border-radius: 6px;
                margin: -3px 0;
            }
        """)

    def set_videos(self, video1: str, video2: str):
        self.view1.set_video(video1)
        self.view2.set_video(video2)