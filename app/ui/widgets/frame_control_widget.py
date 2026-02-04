from PySide6.QtWidgets import (
    QWidget, QFrame, QPushButton, QLabel,
    QHBoxLayout
)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QSize, Signal


def icon_from_svg(svg: str) -> QIcon:
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode("utf-8"))
    return QIcon(pixmap)


CHEVRON_LEFT = """
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
<path d="M15.7 16.3a1 1 0 0 1-1.4 1.4l-6-6a1 1 0 0 1 0-1.4l6-6a1 1 0 0 1 1.4 1.4L10.42 12l5.28 5.3z"/>
</svg>
"""

CHEVRON_RIGHT = """
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
<path d="M8.3 7.7a1 1 0 0 1 1.4-1.4l6 6a1 1 0 0 1 0 1.4l-6 6a1 1 0 0 1-1.4-1.4L13.58 12 8.3 7.7z"/>
</svg>
"""


class FrameControlWidget(QFrame):
    frameChanged = Signal(int)
    reachStart = Signal()
    reachEnd = Signal()

    def __init__(
        self,
        total_frames: int = 1,
        current_frame: int = 1,
        parent=None
    ):
        super().__init__(parent)

        self._total_frames = max(1, total_frames)
        self._current_frame = max(1, min(current_frame, self._total_frames))

        self.setObjectName("FrameControlWidget")
        self.setFixedHeight(52)

        self._build_ui()
        self._connect_signals()
        self._apply_style()
        self._update_ui()

    def current_frame(self) -> int:
        return self._current_frame

    def total_frames(self) -> int:
        return self._total_frames

    def set_total_frames(self, total: int):
        self._total_frames = max(1, total)
        if self._current_frame > self._total_frames:
            self._current_frame = self._total_frames
        self._update_ui()

    def set_current_frame(self, frame: int, emit_signal: bool = True):
        frame = max(1, min(frame, self._total_frames))
        if frame == self._current_frame:
            return

        self._current_frame = frame
        self._update_ui()

        if emit_signal:
            self.frameChanged.emit(self._current_frame)

        if frame == 1:
            self.reachStart.emit()
        elif frame == self._total_frames:
            self.reachEnd.emit()

    def step_prev(self):
        self.set_current_frame(self._current_frame - 1)

    def step_next(self):
        self.set_current_frame(self._current_frame + 1)

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.prev_btn = QPushButton()
        self.prev_btn.setFixedSize(36, 36)
        self.prev_btn.setIcon(icon_from_svg(CHEVRON_LEFT))
        self.prev_btn.setIconSize(QSize(20, 20))

        self.next_btn = QPushButton()
        self.next_btn.setFixedSize(36, 36)
        self.next_btn.setIcon(icon_from_svg(CHEVRON_RIGHT))
        self.next_btn.setIconSize(QSize(20, 20))

        frame_display = QWidget()
        frame_layout = QHBoxLayout(frame_display)
        frame_layout.setContentsMargins(12, 0, 12, 0)
        frame_layout.setSpacing(4)

        self.current_label = QLabel()
        self.current_label.setObjectName("CurrentFrame")

        divider = QLabel("/")
        divider.setObjectName("Divider")

        self.total_label = QLabel()
        self.total_label.setObjectName("TotalFrame")

        frame_layout.addWidget(self.current_label)
        frame_layout.addWidget(divider)
        frame_layout.addWidget(self.total_label)

        layout.addWidget(self.prev_btn)
        layout.addWidget(frame_display)
        layout.addWidget(self.next_btn)

    def _connect_signals(self):
        self.prev_btn.clicked.connect(self.step_prev)
        self.next_btn.clicked.connect(self.step_next)

    def _update_ui(self):
        self.current_label.setText(str(self._current_frame))
        self.total_label.setText(str(self._total_frames))

        self.prev_btn.setDisabled(self._current_frame <= 1)
        self.next_btn.setDisabled(self._current_frame >= self._total_frames)

    def _apply_style(self):
        self.setStyleSheet("""
        #FrameControlWidget {
            background: rgba(255, 255, 255, 0.75);
            border: none;
            border-radius: 12px;
        }

        QPushButton {
            border: none;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.06);
            color: #323130;
        }

        QPushButton:hover:!disabled {
            background: rgba(0, 0, 0, 0.10);
        }

        QPushButton:pressed:!disabled {
            background: rgba(0, 120, 212, 0.85);
            color: white;
        }

        QPushButton:disabled {
            background: rgba(0, 0, 0, 0.03);
            color: rgba(50, 49, 48, 0.35);
        }

        QPushButton:focus-visible {
            outline: none;
            box-shadow: inset 0 0 0 2px rgba(0, 120, 212, 0.6);
        }

        QLabel#CurrentFrame {
            font-size: 16px;
            font-weight: 600;
            color: #0078d4;
        }

        QLabel#Divider {
            font-size: 14px;
            color: rgba(96, 94, 92, 0.45);
        }

        QLabel#TotalFrame {
            font-size: 14px;
            color: rgba(96, 94, 92, 0.8);
        }
        """)