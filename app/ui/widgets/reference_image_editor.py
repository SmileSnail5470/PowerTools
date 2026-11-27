import os
import tempfile
import uuid
from pathlib import Path
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, 
    QSpinBox, QVBoxLayout, QWidget
)
from app.ui.common.config import cfg
from app.ui.library.qfluentwidgets import setFont
from app.ui.widgets.image_preview_widget import ScrollBar


class ReferenceImageCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMouseTracking(True)
        self.image = QImage()
        self.strokes = []
        self.mode = "lasso"
        self.stroke_color = QColor("#ef4444")
        self.brush_size = 24
        self._active_stroke = None
        self._active_cursor = None

    def set_content(self, image: QImage, strokes: list[dict]):
        self.image = image
        self.strokes = strokes
        self._active_stroke = None
        self._active_cursor = None
        self.update()

    def _image_rect(self):
        if self.image.isNull():
            return None
        bounds = self.rect().adjusted(12, 12, -12, -12)
        scale = min(bounds.width() / self.image.width(), bounds.height() / self.image.height())
        width = self.image.width() * scale
        height = self.image.height() * scale
        return QPointF(bounds.left() + (bounds.width() - width) / 2, bounds.top() + (bounds.height() - height) / 2), width, height

    def _to_image_point(self, point: QPointF):
        image_rect = self._image_rect()
        if image_rect is None:
            return None
        origin, width, height = image_rect
        if not (origin.x() <= point.x() <= origin.x() + width and origin.y() <= point.y() <= origin.y() + height):
            return None
        return QPointF((point.x() - origin.x()) * self.image.width() / width, (point.y() - origin.y()) * self.image.height() / height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#f1f5f9"))
        image_rect = self._image_rect()
        if image_rect is None:
            return
        origin, width, height = image_rect
        painter.drawImage(QRectF(origin.x(), origin.y(), width, height), self.image)
        painter.save()
        painter.translate(origin)
        painter.scale(width / self.image.width(), height / self.image.height())
        for stroke in self.strokes:
            self._draw_stroke(painter, stroke)
        if self._active_stroke is not None:
            preview_stroke = dict(self._active_stroke)
            if self._active_cursor is not None:
                preview_stroke["points"] = [*preview_stroke["points"], self._active_cursor]
            self._draw_stroke(painter, preview_stroke)
        painter.restore()

    @staticmethod
    def _draw_stroke(painter: QPainter, stroke: dict):
        if not stroke["points"]:
            return
        path = QPainterPath(stroke["points"][0])
        for point in stroke["points"][1:]:
            path.lineTo(point)
        if stroke["mode"] == "lasso" and len(stroke["points"]) > 2:
            path.closeSubpath()
        pen = QPen(
            QColor(stroke["color"]),
            stroke["width"],
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        point = self._to_image_point(event.position())
        if point is None:
            return
        color = self.stroke_color if self.mode == "lasso" else QColor("#ffffff")
        self._active_stroke = {
            "mode": self.mode,
            "color": color.name(),
            "width": self.brush_size,
            "points": [point],
        }
        self._active_cursor = point
        self.update()

    def mouseMoveEvent(self, event):
        if self._active_stroke is None:
            return
        point = self._clamp_to_image(event.position())
        if point is not None:
            self._active_cursor = point
            self._active_stroke["points"].append(point)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._active_stroke is not None:
            point = self._clamp_to_image(event.position())
            if point is not None and point != self._active_stroke["points"][-1]:
                self._active_stroke["points"].append(point)
            if len(self._active_stroke["points"]) > 1:
                self.strokes.append(self._active_stroke)
            self._active_stroke = None
            self._active_cursor = None
            self.update()

    def _clamp_to_image(self, point: QPointF):
        image_rect = self._image_rect()
        if image_rect is None:
            return None
        origin, width, height = image_rect
        if not (origin.x() <= point.x() <= origin.x() + width and origin.y() <= point.y() <= origin.y() + height):
            point = QPointF(min(max(point.x(), origin.x()), origin.x() + width), min(max(point.y(), origin.y()), origin.y() + height))
        return self._to_image_point(point)


class ReferenceImageEditorDialog(QDialog):
    def __init__(self, file_paths: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("编辑参考图"))
        self.resize(1220, 800)
        self.setMinimumSize(1050, 680)
        self.setStyleSheet("QDialog { background: #f8fafc; }")
        self._states = []
        self.updated_paths = list(file_paths)
        for path in file_paths:
            image = QImage(path)
            if image.isNull():
                raise ValueError(self.tr("无法读取参考图：{0}").format(path))
            self._states.append({"path": path, "image": image, "strokes": []})
        self._selected_index = 0
        self._buttons = []
        self._build_ui()
        self._select_image(0)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(12)
        workspace = QHBoxLayout()
        workspace.setSpacing(14)
        layout.addLayout(workspace, 1)

        sidebar = QFrame(self)
        sidebar.setFixedWidth(285)
        sidebar.setStyleSheet("QFrame { background: white; border: 1px solid #e2e8f0; border-radius: 10px; }")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(10)
        title = QLabel(self.tr("参考图"))
        setFont(title, 12, QFont.DemiBold)
        title.setStyleSheet("QLabel { color: #1e293b; background: transparent; border: none; }")
        sidebar_layout.addWidget(title)
        scroll = QScrollArea(sidebar)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(285)
        _v_scroll = ScrollBar(Qt.Vertical, self)
        _h_scroll = ScrollBar(Qt.Horizontal, self)
        scroll.setVerticalScrollBar(_v_scroll)
        scroll.setHorizontalScrollBar(_h_scroll)
        scroll.setStyleSheet("""QScrollArea { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 7px; }""")
        items = QWidget(scroll)
        items.setObjectName("referenceThumbnailList")
        items.setAutoFillBackground(True)
        items.setStyleSheet("QWidget#referenceThumbnailList { background-color: #f1f5f9; color: #334155; }")
        self._thumbnail_layout = QVBoxLayout(items)
        self._thumbnail_layout.setContentsMargins(7, 7, 7, 7)
        self._thumbnail_layout.setSpacing(8)
        for index, state in enumerate(self._states):
            button = QPushButton(items)
            button.setMinimumHeight(112)
            button.setCheckable(True)
            button.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 5px;
                    color: #1e293b;
                    background-color: #f8fafc;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                }
                QPushButton:hover { background-color: #eef2f7; }
                QPushButton:checked {
                    color: #1d4ed8;
                    background-color: #dbeafe;
                    border: 1px solid #3b82f6;
                }
            """)
            row = QHBoxLayout(button)
            row.setContentsMargins(7, 6, 7, 6)
            thumbnail = QLabel(button)
            thumbnail.setFixedSize(220, 96)
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail.setStyleSheet(
                "QLabel { background-color: #e2e8f0; border: 1px solid #cbd5e1; "
                "border-radius: 4px; }"
            )
            thumbnail.setPixmap(QPixmap.fromImage(state["image"]).scaled(
                thumbnail.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
            row.addWidget(thumbnail)
            button.clicked.connect(lambda checked=False, item_index=index: self._select_image(item_index))
            self._buttons.append(button)
            self._thumbnail_layout.addWidget(button)
        self._thumbnail_layout.addStretch(1)
        scroll.setWidget(items)
        sidebar_layout.addWidget(scroll, 1)

        settings_title = QLabel(self.tr("功能设置"))
        setFont(settings_title, 12, QFont.DemiBold)
        settings_title.setStyleSheet("QLabel { color: #1e293b; background: transparent; border: none; }")
        sidebar_layout.addWidget(settings_title)

        mode_label = QLabel(self.tr("任务类型"))
        mode_label.setStyleSheet("QLabel { color: #334155; background: transparent; border: none; }")
        sidebar_layout.addWidget(mode_label)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(7)
        self.lasso_button = QPushButton(self.tr("圈选"))
        self.brush_button = QPushButton(self.tr("白色涂抹"))
        for button in (self.lasso_button, self.brush_button):
            button.setCheckable(True)
            button.setMinimumHeight(34)
            button.setStyleSheet(self._tool_button_style())
            mode_row.addWidget(button, 1)
        self.lasso_button.setChecked(True)
        self.lasso_button.clicked.connect(self._set_lasso_mode)
        self.brush_button.clicked.connect(self._set_brush_mode)
        sidebar_layout.addLayout(mode_row)
        self._mode_label = mode_label

        color_label = QLabel(self.tr("圈选线颜色"))
        color_label.setStyleSheet("QLabel { color: #334155; background: transparent; border: none; }")
        sidebar_layout.addWidget(color_label)
        self._color_buttons = []
        self._color_values = (
            ("红色", "#ef4444"),
            ("橙色", "#f97316"),
            ("黄色", "#eab308"),
            ("绿色", "#22c55e"),
            ("青色", "#06b6d4"),
            ("蓝色", "#3b82f6"),
            ("紫色", "#8b5cf6"),
            ("黑色", "#111827"),
        )
        color_grid = QGridLayout()
        color_grid.setContentsMargins(2, 6, 2, 6)
        color_grid.setHorizontalSpacing(12)
        color_grid.setVerticalSpacing(20)
        for index, (color_name, color_value) in enumerate(self._color_values):
            color_button = QPushButton()
            color_button.setCheckable(True)
            color_button.setFixedSize(32, 32)
            color_button.setToolTip(self.tr(color_name))
            color_button.clicked.connect(lambda checked=False, selected=color_value: self._select_color(selected))
            self._color_buttons.append((color_value, color_button))
            color_grid.addWidget(color_button, index // 4, index % 4)
        sidebar_layout.addLayout(color_grid)
        self._color_label = color_label
        self._color_grid = color_grid

        brush_row = QHBoxLayout()
        brush_label = QLabel(self.tr("画笔大小"))
        brush_label.setStyleSheet("QLabel { color: #334155; background: transparent; border: none; }")
        self.brush_size_label = brush_label
        brush_row.addWidget(brush_label)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(2, 200)
        self.brush_size.setValue(24)
        self.brush_size.setSuffix(" px")
        self.brush_size.setObjectName("frameRangeInput")
        self.brush_size.setFixedHeight(30)
        self.brush_size.setStyleSheet("""
            QSpinBox {
                padding: 0 18px 0 10px;
                border-radius: 10px;
                color: #1e293b;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                font-size: 12px;
                selection-background-color: #6366f1;
            }
            QSpinBox:hover {
                background: #fff;
                border-color: #cbd5e1;
            }
            QSpinBox:focus {
                background: #fff;
                border-color: #818cf8;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                subcontrol-origin: border;
                width: 13px;
                height: 12px;
                border: none;
                background: transparent;
            }
            QSpinBox::up-button {
                subcontrol-position: top right;
                top: 2px;
                right: 2px;
            }
            QSpinBox::down-button {
                subcontrol-position: bottom right;
                bottom: 2px;
                right: 2px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: rgba(99, 102, 241, 0.08);
                border-radius: 6px;
            }
            QSpinBox::up-arrow {
                image: url(:/qfluentwidgets/images/icons/CareUpSolid_black.svg);
                width: 7px;
                height: 7px;
            }
            QSpinBox::down-arrow {
                image: url(:/qfluentwidgets/images/icons/CareDownSolid_black.svg);
                width: 7px;
                height: 7px;
            }
        """)
        self.brush_size.valueChanged.connect(self._set_brush_size)
        brush_row.addWidget(self.brush_size)
        sidebar_layout.addLayout(brush_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(7)
        undo_button = QPushButton(self.tr("撤销"))
        undo_button.setStyleSheet(self._tool_button_style())
        undo_button.clicked.connect(self._undo)
        clear_button = QPushButton(self.tr("清除标注"))
        clear_button.setStyleSheet(self._tool_button_style())
        clear_button.clicked.connect(self._clear_current)
        action_row.addWidget(undo_button)
        action_row.addWidget(clear_button)
        sidebar_layout.addLayout(action_row)
        workspace.addWidget(sidebar)

        editor = QFrame(self)
        editor.setStyleSheet("QFrame { background: white; border: 1px solid #e2e8f0; border-radius: 10px; }")
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(12, 12, 12, 12)
        editor_layout.setSpacing(10)
        self.current_name = QLabel()
        setFont(self.current_name, 11, QFont.DemiBold)
        self.current_name.setStyleSheet("QLabel { color: #334155; background: transparent; border: none; }")
        editor_layout.addWidget(self.current_name)
        self.canvas = ReferenceImageCanvas(editor)
        self._select_color(self.canvas.stroke_color.name())
        editor_layout.addWidget(self.canvas, 1)
        workspace.addWidget(editor, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        cancel_button = QPushButton(self.tr("取消"))
        cancel_button.setStyleSheet(self._tool_button_style())
        cancel_button.setAutoDefault(False)
        cancel_button.clicked.connect(self.reject)
        save_button = QPushButton(self.tr("保存编辑"))
        save_button.setStyleSheet("""
            QPushButton {
                color: #ffffff;
                background-color: #2563eb;
                border: 1px solid #1d4ed8;
                border-radius: 6px;
                padding: 7px 16px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:pressed { background-color: #1e40af; }
        """)
        save_button.setAutoDefault(False)
        save_button.clicked.connect(self._save)
        footer.addWidget(cancel_button)
        footer.addWidget(save_button)
        layout.addLayout(footer)

    @staticmethod
    def _tool_button_style():
        return """
            QPushButton {
                color: #334155;
                background-color: #f1f5f9;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QPushButton:hover { background-color: #e2e8f0; }
            QPushButton:pressed { background-color: #cbd5e1; }
            QPushButton:checked {
                color: #1d4ed8;
                background-color: #dbeafe;
                border: 1px solid #60a5fa;
            }
            QPushButton:disabled {
                color: #94a3b8;
                background-color: #f8fafc;
                border-color: #e2e8f0;
            }
        """

    def _select_image(self, index: int):
        self._selected_index = index
        state = self._states[index]
        self.current_name.setText(self.tr("文件名：{0}").format(os.path.basename(state["path"])))
        self.canvas.set_content(state["image"], state["strokes"])
        for button_index, button in enumerate(self._buttons):
            button.setChecked(button_index == index)

    def _set_lasso_mode(self):
        self.canvas.mode = "lasso"
        self.brush_size_label.setText(self.tr("画笔大小"))
        self.brush_button.setChecked(False)
        self.lasso_button.setChecked(True)
        self._color_label.setEnabled(True)
        self._color_label.show()
        for _, button in self._color_buttons:
            button.setEnabled(True)
            button.show()

    def _set_brush_mode(self):
        self.canvas.mode = "brush"
        self.brush_size_label.setText(self.tr("画刷大小"))
        self.lasso_button.setChecked(False)
        self.brush_button.setChecked(True)
        self._color_label.setEnabled(False)
        self._color_label.hide()
        for _, button in self._color_buttons:
            button.setEnabled(False)
            button.hide()

    def _select_color(self, color_value: str):
        self.canvas.stroke_color = QColor(color_value)
        for value, button in self._color_buttons:
            border = "#1e293b" if value == color_value else "#cbd5e1"
            border_width = 3 if value == color_value else 1
            button.setChecked(value == color_value)
            button.setStyleSheet(
                "QPushButton { background-color: %s; border: %dpx solid %s; "
                "border-radius: 15px; } QPushButton:hover { border-color: #0f172a; }"
                % (value, border_width, border)
            )

    def _set_brush_size(self, size: int):
        self.canvas.brush_size = size

    def _undo(self):
        if self.canvas.strokes:
            self.canvas.strokes.pop()
            self.canvas.update()

    def _clear_current(self):
        self.canvas.strokes.clear()
        self.canvas.update()

    def _save(self):
        cache_root = cfg.get(cfg.cachePath) or tempfile.gettempdir()
        output_dir = Path(cache_root) / "image_edit" / "reference_edits"
        created_files = []
        updated_paths = list(self.updated_paths)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            for index, state in enumerate(self._states):
                if not state["strokes"]:
                    continue
                source_path = Path(state["path"])
                output_path = output_dir / (f"{source_path.stem}_{uuid.uuid4().hex}{source_path.suffix or '.png'}")
                if not self._render_state(state).save(str(output_path)):
                    raise OSError(self.tr("无法保存编辑后的参考图：{0}").format(output_path))
                created_files.append(output_path)
                updated_paths[index] = str(output_path)
        except OSError as exc:
            for path in created_files:
                path.unlink(missing_ok=True)
            QMessageBox.warning(self, self.tr("保存失败"), str(exc))
            return
        self.updated_paths = updated_paths
        self.accept()

    @staticmethod
    def _render_state(state: dict) -> QImage:
        image = state["image"].convertToFormat(QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        for stroke in state["strokes"]:
            if not stroke["points"]:
                continue
            path = QPainterPath(stroke["points"][0])
            for point in stroke["points"][1:]:
                path.lineTo(point)
            if stroke["mode"] == "lasso" and len(stroke["points"]) > 2:
                path.closeSubpath()
            painter.setPen(QPen(
                QColor(stroke["color"]),
                stroke["width"],
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            ))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
        painter.end()
        return image
