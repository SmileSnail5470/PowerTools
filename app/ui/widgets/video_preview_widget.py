import os
from PySide6.QtCore import Signal, QTimer, Slot, QUrl
from PySide6.QtWidgets import QHBoxLayout, QWidget , QVBoxLayout
from app.ui.library.qfluentwidgets.multimedia import VideoWidget, SimpleMediaPlayBar


class SyncVideoViewer(QWidget):
    playbackStateChanged = Signal(bool)
    positionChanged = Signal(int)

    def __init__(self, parent=None, sync_interval=200, drift_threshold=150):
        super().__init__(parent=parent)
        self.videos = []
        self.master = None
        self.sync_interval = sync_interval
        self.drift_threshold = drift_threshold

        self._create_ui()
        self._connect_signals()

        # 定时同步器
        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(self.sync_interval)
        self.sync_timer.timeout.connect(self._sync_videos)

        self.add_video(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "videos", "mov_bbb.mp4"))
        self.add_video(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "videos", "mov_bbb.mp4"))

    def _create_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        self.video_layout = QVBoxLayout()
        self.play_bar = SimpleMediaPlayBar()

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(self.play_bar)

        main_layout.addLayout(self.video_layout, 1)
        main_layout.addLayout(ctrl_layout, 0)

    def _connect_signals(self):
        self.play_bar.playButton.clicked.connect(self.toggle_play)
        self.play_bar.progressSlider.sliderMoved.connect(self._on_slider_moved)

    def add_video(self, path: str):
        video = VideoWidget(self)
        video.playBar.hide()
        video.setVideo(QUrl.fromLocalFile(path))

        video.player.durationChanged.connect(self._on_duration_changed)
        video.player.positionChanged.connect(self._on_position_changed)

        self.videos.append(video)
        self.video_layout.addWidget(video)

        if not self.master:
            self.master = video

    @Slot()
    def play(self):
        if not self.videos:
            return
        for v in self.videos:
            v.play()
        self.sync_timer.start()
        self.playbackStateChanged.emit(True)
        self.play_bar.playButton.setPlay(True)

    @Slot()
    def pause(self):
        for v in self.videos:
            v.pause()
        self.sync_timer.stop()
        self.playbackStateChanged.emit(False)
        self.play_bar.playButton.setPlay(False)

    @Slot()
    def stop(self):
        for v in self.videos:
            v.stop()
        self.sync_timer.stop()
        self.play_bar.playButton.setPlay(False)
        self.play_bar.progressSlider.setValue(0)

    @Slot()
    def toggle_play(self):
        if not self.master:
            return
        if self.master.player.isPlaying():
            self.pause()
        else:
            self.play()

    @Slot(int)
    def _on_slider_moved(self, value: int):
        if not self.master:
            return
        duration = self.master.player.duration()
        if duration > 0:
            pos = int(value / 100 * duration)
            for v in self.videos:
                v.player.setPosition(pos)

    def _on_position_changed(self, pos: int):
        if self.sender() != self.master.player:
            return
        duration = self.master.player.duration()
        if duration > 0:
            percent = int(pos / duration * 100)
            self.play_bar.progressSlider.blockSignals(True)
            self.play_bar.progressSlider.setValue(percent)
            self.play_bar.progressSlider.blockSignals(False)
        self.positionChanged.emit(pos)

    def _on_duration_changed(self, duration: int):
        if self.sender() == self.master.player:
            self.play_bar.progressSlider.setRange(0, 100)

    def _sync_videos(self):
        if not self.master or not self.master.player.isPlaying():
            return
        master_pos = self.master.player.position()
        for v in self.videos:
            if v is self.master:
                continue
            diff = abs(v.player.position() - master_pos)
            if diff > self.drift_threshold:
                v.player.setPosition(master_pos)

    def setPlaybackRate(self, rate: float):
        for v in self.videos:
            v.player.setPlaybackRate(rate)

    def _format_time(self, ms: int) -> str:
        s = int(ms / 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"