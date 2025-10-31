import os
from PySide6.QtCore import Qt, Signal, QUrl, QSizeF, QTimer, Slot
from PySide6.QtGui import QPainter, QColor, QBrush
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QHBoxLayout, QVBoxLayout, QWidget, QGraphicsRectItem

from app.ui.library.qfluentwidgets.common.style_sheet import FluentStyleSheet, isDarkTheme
from app.ui.library.qfluentwidgets.multimedia.media_play_bar import MediaPlayer, MediaPlayerBase, VolumeButton, PlayButton
from app.ui.library.qfluentwidgets import Slider


class MediaPlayBarBase(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.players = []

        self.playButton = PlayButton(self)
        self.volumeButton = VolumeButton(self)
        self.progressSlider = Slider(Qt.Horizontal, self)

        FluentStyleSheet.MEDIA_PLAYER.apply(self)

        self.playButton.clicked.connect(self.togglePlayState)

    def setMediaPlayer(self, player: MediaPlayerBase):
        if not self.players:
            player.durationChanged.connect(self.progressSlider.setMaximum)
            player.positionChanged.connect(self._onPositionChanged)
            player.mediaStatusChanged.connect(self._onMediaStatusChanged)
            player.volumeChanged.connect(self.volumeButton.setVolume)
            player.mutedChanged.connect(self.volumeButton.setMuted)

        self.progressSlider.sliderMoved.connect(player.setPosition)
        self.progressSlider.clicked.connect(player.setPosition)
        self.volumeButton.volumeChanged.connect(player.setVolume)
        self.volumeButton.mutedChanged.connect(player.setMuted)

        player.setVolume(30)
        
        self.players.append(player)

    def play(self):
        for player in self.players:
            player.play()

    def pause(self):
        for player in self.players:
            player.pause()

    def stop(self):
        for player in self.players:
            player.stop()

    def setVolume(self, volume: int):
        for player in self.players:
            player.setVolume(volume)

    def setPosition(self, position: int):
        for player in self.players:
            player.setPosition(position)

    def _onPositionChanged(self, position: int):
        self.progressSlider.setValue(position)

    def _onMediaStatusChanged(self, status):
        self.playButton.setPlay(self.players[0].isPlaying())

    def togglePlayState(self):
        if self.players[0].isPlaying():
            for player in self.players:
                player.pause()
        else:
            for player in self.players:
                player.play()

        self.playButton.setPlay(self.players[0].isPlaying())

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        if isDarkTheme():
            painter.setBrush(QColor(46, 46, 46))
            painter.setPen(QColor(0, 0, 0, 20))
        else:
            painter.setBrush(QColor(248, 248, 248))
            painter.setPen(QColor(0, 0, 0, 10))

        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)

class CustomMediaPlayBar(MediaPlayBarBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hBoxLayout = QHBoxLayout(self)

        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.hBoxLayout.setSpacing(6)
        self.hBoxLayout.addWidget(self.playButton, 0, Qt.AlignLeft)
        self.hBoxLayout.addWidget(self.progressSlider, 1)
        self.hBoxLayout.addWidget(self.volumeButton, 0)

        self.setFixedHeight(48)


class SyncVideoViewer(QWidget):
    playbackStateChanged = Signal(bool)
    positionChanged = Signal(int)

    def __init__(self, parent=None, sync_interval=200, drift_threshold=150):
        super().__init__(parent)
        self.sync_interval = sync_interval
        self.drift_threshold = drift_threshold

        self.player_main = MediaPlayer(self)
        self.player_sub = MediaPlayer(self)

        self.video_main = QGraphicsVideoItem()
        self.video_sub = QGraphicsVideoItem()

        self.player_main.setVideoOutput(self.video_main)
        self.player_sub.setVideoOutput(self.video_sub)

        self.scene = QGraphicsScene(self)
        # 背景矩形（使区域可见）
        self.bg_main = QGraphicsRectItem()
        self.bg_main.setBrush(QBrush(QColor(120, 120, 120)))
        self.bg_main.setPen(Qt.NoPen)
        self.scene.addItem(self.bg_main)

        self.bg_sub = QGraphicsRectItem()
        self.bg_sub.setBrush(QBrush(QColor(100, 100, 100)))
        self.bg_sub.setPen(Qt.NoPen)
        self.scene.addItem(self.bg_sub)

        self.scene.addItem(self.video_main)
        self.scene.addItem(self.video_sub)

        # 分割线
        self.divider = QGraphicsRectItem()
        self.divider.setBrush(QBrush(QColor(180, 180, 180)))
        self.divider.setPen(Qt.NoPen)
        self.scene.addItem(self.divider)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        FluentStyleSheet.MEDIA_PLAYER.apply(self.view)

        self.playBar = CustomMediaPlayBar(self)
        self.playBar.setMediaPlayer(self.player_main)
        self.playBar.setMediaPlayer(self.player_sub)

        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(self._sync_videos)
        self.sync_timer.start(self.sync_interval)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.playBar, 0)

        self.setVideos(
            QUrl.fromLocalFile(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "videos", "mov_bbb.mp4")),
            QUrl.fromLocalFile(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "videos", "mov_bbb.mp4"))
        )

    def setVideos(self, main_url: QUrl, sub_url: QUrl):
        self.player_main.setSource(main_url)
        self.player_sub.setSource(sub_url)
        self._updateVideoLayout()

    def _updateVideoLayout(self):
        w, h = self.view.width(), self.view.height() / 2

        self.bg_main.setRect(0, 0, w, h - 1)
        self.bg_sub.setRect(0, h + 1, w, h - 1)

        self.video_main.setSize(QSizeF(w, h - 1))
        self.video_sub.setSize(QSizeF(w, h - 1))
        self.video_main.setPos(0, 0)
        self.video_sub.setPos(0, h + 1)

        self.divider.setRect(0, h - 1, w, 2)

        self.scene.setSceneRect(0, 0, w, h * 2)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._updateVideoLayout()

    def showEvent(self, event):
        super().showEvent(event)
        self._updateVideoLayout()

    @Slot()
    def _sync_videos(self):
        if self.player_main.isPlaying() and self.player_sub.isPlaying():
            pos_main = self.player_main.position()
            pos_sub = self.player_sub.position()
            if abs(pos_main - pos_sub) > self.drift_threshold:
                self.player_sub.setPosition(pos_main)