import os
import sys
from PySide6.QtCore import QRegularExpression, QSize, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QPixmap, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QAbstractScrollArea, QFileDialog, QButtonGroup, QDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QStackedWidget, QTextEdit, QVBoxLayout, QWidget
)
from app.ui.common.event_bus import global_event_bus
from app.ui.common.task_params import LayeredTaskParams
from app.ui.library.qfluentwidgets import FluentIcon, setFont
from app.ui.widgets.reference_image_editor import ReferenceImageEditorDialog
from app.ui.widgets.image_preview_widget import ScrollBar


def _card_colors():
    return {
        "card": "#f8f9fa",
        "sub": "#f8f9fa",
        "elevated": "#f8f9fa",
        "border": "rgba(15, 23, 42, 0.08)",
        "strong_border": "rgba(15, 23, 42, 0.13)",
        "primary": "#0f172a",
        "secondary": "#475569",
        "muted": "#94a3b8",
        "accent": "#0067c0",
        "accent_hover": "#005a9e",
        "tint": "#eff6ff",
        "drop": "#f8f9fa",
    }


class SettingsSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)


class SectionHeading(QWidget):
    def __init__(self, icon: str, title: str, description: str = "", parent=None):
        super().__init__(parent)
        colors = _card_colors()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)

        icon_label = QLabel(icon, self)
        icon_label.setFixedSize(28, 28)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("background: %(tint)s; color: %(accent)s; border-radius: 9px; font-size: 15px;" % colors)
        layout.addWidget(icon_label, 0, Qt.AlignTop)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        title_label = QLabel(title, self)
        setFont(title_label, 13, QFont.DemiBold)
        title_label.setStyleSheet("color: %s; background: transparent;" % colors["primary"])
        text_layout.addWidget(title_label)
        if description:
            description_label = QLabel(description, self)
            description_label.setWordWrap(True)
            setFont(description_label, 10)
            description_label.setStyleSheet("color: %s; background: transparent;" % colors["muted"])
            text_layout.addWidget(description_label)
        layout.addLayout(text_layout, 1)


class PromptEditorCard(SettingsSection):
    def __init__(
        self,
        params: LayeredTaskParams,
        parent=None,
        prompt_path: str = "task.prompt",
        max_characters: int = 600,
        placeholder: str = "描述你希望如何编辑图片，可使用自然语言说明主体、风格与细节。",
    ):
        super().__init__(parent)
        self.params = params
        self.prompt_path = prompt_path
        self.max_characters = max_characters

        heading_row = QHBoxLayout()
        heading_row.setContentsMargins(0, 0, 0, 0)
        heading_row.addWidget(SectionHeading("✎", "编辑提示词", parent=self), 1)

        colors = _card_colors()
        self.prompt_edit = QTextEdit(self)
        self.prompt_edit.setPlaceholderText(placeholder)
        self.prompt_edit.setPlainText(params.get_param(prompt_path))
        self.prompt_edit.setMinimumHeight(98)
        self.prompt_edit.setMaximumHeight(132)
        _v_scroll = ScrollBar(Qt.Vertical, self)
        _h_scroll = ScrollBar(Qt.Horizontal, self)
        self.prompt_edit.setVerticalScrollBar(_v_scroll)
        self.prompt_edit.setHorizontalScrollBar(_h_scroll)
        setFont(self.prompt_edit, 12)
        self.prompt_edit.setStyleSheet("""
            QTextEdit {
                color: %(primary)s;
                background: %(elevated)s;
                border: 1px solid %(strong_border)s;
                border-radius: 11px;
                padding: 10px;
                selection-background-color: %(accent)s;
            }
            QTextEdit:focus {
                border: 1px solid %(accent)s;
            }
        """ % colors)
        self.prompt_edit.textChanged.connect(self._on_prompt_changed)

        self.clear_button = QPushButton("清空", self)
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.setStyleSheet(self._text_button_style())
        self.clear_button.clicked.connect(self.prompt_edit.clear)
        heading_row.addWidget(self.clear_button, 0, Qt.AlignTop)
        self.content_layout.addLayout(heading_row)

        self.content_layout.addWidget(self.prompt_edit)

        footer = QHBoxLayout()
        footer.setContentsMargins(2, 0, 2, 0)
        self.character_count = QLabel(self)
        setFont(self.character_count, 10)
        self.character_count.setStyleSheet("color: %s;" % colors["muted"])
        footer.addWidget(self.character_count)
        footer.addStretch(1)
        hint = QLabel("支持中文与英文自然语言", self)
        setFont(hint, 10)
        hint.setStyleSheet("color: %s;" % colors["muted"])
        footer.addWidget(hint)
        self.content_layout.addLayout(footer)
        self._update_character_count()

    @staticmethod
    def _text_button_style():
        colors = _card_colors()
        return """
            QPushButton {
                color: %(muted)s;
                background: transparent;
                border: none;
                border-radius: 7px;
                padding: 5px 8px;
            }
            QPushButton:hover {
                color: #e11d48;
                background: %(tint)s;
            }
        """ % colors

    def _on_prompt_changed(self):
        text = self.prompt_edit.toPlainText()
        self.params.set_param(self.prompt_path, text)
        self._update_character_count()

    def _update_character_count(self):
        count = len(self.prompt_edit.toPlainText())
        self.character_count.setText(f"{count} / {self.max_characters} 字")


class ReferenceDropZone(QFrame):
    files_dropped = Signal(list)

    def __init__(self, choose_file, parent=None):
        super().__init__(parent)
        self.choose_file = choose_file
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(96)
        self._set_drag_style(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        icon = QLabel("+", self)
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedSize(30, 30)
        setFont(icon, 22, QFont.Light)
        colors = _card_colors()
        icon.setStyleSheet("color: %(accent)s; background: %(elevated)s; border-radius: 15px;" % colors)
        layout.addWidget(icon, 0, Qt.AlignHCenter)
        title_label = QLabel("添加参考图", self)
        setFont(title_label, 11, QFont.DemiBold)
        title_label.setStyleSheet("color: %s;" % colors["accent"])
        layout.addWidget(title_label, 0, Qt.AlignHCenter)
        self.hint_label = QLabel("支持多选或拖拽多张参考图片", self)
        setFont(self.hint_label, 9)
        self.hint_label.setStyleSheet("color: %s;" % colors["muted"])
        layout.addWidget(self.hint_label, 0, Qt.AlignHCenter)
        for label in (icon, title_label, self.hint_label):
            label.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_status(self, text: str, is_error: bool = False):
        colors = _card_colors()
        color = "#e11d48" if is_error else colors["muted"]
        self.hint_label.setText(text)
        self.hint_label.setStyleSheet("color: %s;" % color)

    def _set_drag_style(self, active):
        colors = _card_colors()
        background = colors["tint"] if active else colors["drop"]
        self.setStyleSheet("""
            QFrame {
                background: %(background)s;
                border: 1.5px dashed %(accent)s;
                border-radius: 11px;
            }
            QFrame:hover {
                background: %(tint)s;
                border-color: %(accent_hover)s;
            }
            QLabel { background: transparent; border: none; }
        """ % {**colors, "background": background})

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.choose_file()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_drag_style(True)

    def dragLeaveEvent(self, event):
        self._set_drag_style(False)

    def dropEvent(self, event):
        self._set_drag_style(False)
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


class ReferenceImageCard(SettingsSection):
    IMAGE_FILTER = "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tiff)"
    MAX_REFERENCE_IMAGES = 10

    def __init__(
        self,
        params: LayeredTaskParams,
        parent=None,
        input_path: str = "io.input_path",
        mask_path: str = "task.mask_boxes",
        reference_paths: str = "task.reference_paths",
        action_label: str = "圈选或涂抹重绘区域",
    ):
        super().__init__(parent)
        self.params = params
        self.input_path = input_path
        self.mask_path = mask_path
        self.reference_paths = reference_paths
        self.action_label = action_label
        self._file_paths = list(params.get_param(reference_paths) or [])
        if not self._file_paths and params.get_param(input_path):
            self._file_paths = [params.get_param(input_path)]

        self.content_layout.addWidget(SectionHeading("▧", "参考图", parent=self))

        self.drop_zone = ReferenceDropZone(self._choose_file, self)
        self.drop_zone.files_dropped.connect(self._add_files)
        self.content_layout.addWidget(self.drop_zone)

        self.file_grid_host = QWidget(self)
        self.file_grid = QGridLayout(self.file_grid_host)
        self.file_grid.setContentsMargins(0, 0, 0, 0)
        self.file_grid.setSpacing(7)
        self.content_layout.addWidget(self.file_grid_host)

        colors = _card_colors()
        self.area_info = QLabel("尚未选择重绘区域", self)
        setFont(self.area_info, 9)
        self.area_info.setStyleSheet("color: %s;" % colors["muted"])
        self.content_layout.addWidget(self.area_info)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        self.area_select_button = QPushButton(self.action_label, self)
        setFont(self.area_select_button, 10)
        self.area_select_button.setCursor(Qt.PointingHandCursor)
        self.area_select_button.setStyleSheet(self._action_button_style())
        self.area_select_button.clicked.connect(self._select_area)
        self.area_select_button.setEnabled(False)
        actions.addWidget(self.area_select_button)
        self.clear_button = QPushButton("清空参考图", self)
        setFont(self.clear_button, 10)
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.setStyleSheet(PromptEditorCard._text_button_style())
        self.clear_button.clicked.connect(lambda: self._set_files([]))
        actions.addWidget(self.clear_button)
        actions.addStretch(1)
        self.content_layout.addLayout(actions)
        self._render_files()

    @staticmethod
    def _action_button_style():
        colors = _card_colors()
        return """
            QPushButton {
                color: %(accent)s;
                background: %(tint)s;
                border: 1px solid %(border)s;
                border-radius: 8px;
                padding: 6px 9px;
            }
            QPushButton:hover { background: %(tint)s; }
            QPushButton:disabled { color: %(muted)s; }
        """ % colors

    def _choose_file(self):
        options = (QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0))
        paths, _ = QFileDialog.getOpenFileNames(self, self.tr("选择参考图片（可多选）"), "", self.IMAGE_FILTER, options=options)
        if paths:
            self._add_files(paths)

    def _add_files(self, file_paths: list[str]):
        valid_paths = []
        invalid_paths = []
        for path in file_paths:
            if not os.path.isfile(path) or QPixmap(path).isNull():
                invalid_paths.append(path)
            elif path not in self._file_paths and path not in valid_paths:
                valid_paths.append(path)

        available_slots = max(0, self.MAX_REFERENCE_IMAGES - len(self._file_paths))
        accepted_paths = valid_paths[:available_slots]
        exceeded_count = len(valid_paths) - len(accepted_paths)
        if accepted_paths:
            self._set_files([*self._file_paths, *accepted_paths])
        if invalid_paths:
            self.drop_zone.set_status(self.tr("有 {0} 个文件无法读取，已忽略").format(len(invalid_paths)), is_error=True,)
        else:
            self.drop_zone.set_status(self.tr("支持多选或拖拽多张参考图片"))
        if exceeded_count:
            QMessageBox.warning(
                self,
                self.tr("参考图数量已达上限"),
                self.tr("最多只能添加 {0} 张参考图，超出的图片已忽略。").format(self.MAX_REFERENCE_IMAGES),
            )

    def _set_files(self, file_paths: list[str], emit_event=True):
        self._file_paths = list(file_paths)
        primary_path = self._file_paths[0] if self._file_paths else ""
        self.params.set_param(self.reference_paths, self._file_paths)
        self.params.set_param(self.input_path, primary_path)
        self.params.set_param(self.mask_path, [])
        self.area_info.setText(self.tr("尚未选择重绘区域"))
        self.area_select_button.setEnabled(bool(primary_path))
        self._render_files()
        if emit_event:
            global_event_bus.imageEdit_InputFileUpdate.emit(primary_path)

    def _render_files(self):
        while self.file_grid.count():
            item = self.file_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.file_grid_host.setVisible(bool(self._file_paths))
        colors = _card_colors()
        for index, path in enumerate(self._file_paths):
            tile = QFrame(self.file_grid_host)
            tile.setObjectName("referenceTile")
            tile.setStyleSheet("""
                QFrame#referenceTile {
                    background: %(sub)s;
                    border: 1px solid %(border)s;
                    border-radius: 9px;
                }
                QLabel { background: transparent; border: none; }
            """ % colors)
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(6, 6, 6, 6)
            tile_layout.setSpacing(4)
            badge = QLabel(self.tr("参考图 {0}").format(index + 1), tile)
            badge.setAlignment(Qt.AlignHCenter)
            badge.setStyleSheet("color: %s; background: %s; border-radius: 5px; padding: 2px;" % (colors["accent"], colors["tint"]))
            tile_layout.addWidget(badge)
            thumbnail = QLabel(tile)
            thumbnail.setFixedSize(72, 60)
            thumbnail.setAlignment(Qt.AlignCenter)
            thumbnail.setPixmap(QPixmap(path).scaled(thumbnail.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            thumbnail.setToolTip(os.path.basename(path))
            tile_layout.addWidget(thumbnail, 0, Qt.AlignHCenter)
            remove_button = QPushButton("移除", tile)
            remove_button.setCursor(Qt.PointingHandCursor)
            remove_button.setStyleSheet(PromptEditorCard._text_button_style())
            remove_button.clicked.connect(lambda checked=False, item_path=path: self._remove_file(item_path))
            tile_layout.addWidget(remove_button)
            self.file_grid.addWidget(tile, index // 3, index % 3)
        self.clear_button.setEnabled(bool(self._file_paths))

    def _remove_file(self, path: str):
        self._set_files([item for item in self._file_paths if item != path])

    def _select_area(self):
        if not self._file_paths:
            return
        try:
            dialog = ReferenceImageEditorDialog(self._file_paths, parent=self.window())
        except ValueError as exc:
            QMessageBox.warning(self, self.tr("无法编辑参考图"), str(exc))
            return
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        updated_paths = dialog.updated_paths
        edited_count = sum(old_path != new_path for old_path, new_path in zip(self._file_paths, updated_paths))
        if edited_count:
            self._set_files(updated_paths)
            self.area_info.setText(self.tr("{0} 张参考图已保存至缓存").format(edited_count))


def _create_edit_mode_selector(parent, reference_card):
    colors = _card_colors()
    selector = QWidget(parent)
    layout = QVBoxLayout(selector)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(SectionHeading("◇", "任务类型", parent=selector))

    cards_container = QFrame(selector)
    cards_container.setObjectName("taskTypeCardsContainer")
    cards_container.setStyleSheet("""
        QFrame#taskTypeCardsContainer {
            background: #f1f3f5;
            border: none;
            border-radius: 12px;
        }
    """)
    cards_layout = QHBoxLayout(cards_container)
    cards_layout.setContentsMargins(14, 10, 14, 10)
    cards_layout.setSpacing(8)
    group = QButtonGroup(selector)
    group.setExclusive(True)
    mode_buttons = {}
    modes = (
        ("image_edit", "图像编辑", "基于参考图编辑", FluentIcon.EDIT, "#2563eb", "#eff6ff", "#dbeafe"),
        ("text_to_image", "文生图", "根据文字生成图像", FluentIcon.PHOTO, "#7c3aed", "#f5f3ff", "#ede9fe"),
    )
    for mode, title, description, icon, icon_color, tint_color, checked_border_color in modes:
        button = QPushButton(selector)
        button.setCheckable(True)
        button.setMinimumHeight(60)
        button.setCursor(Qt.PointingHandCursor)
        button.setProperty("tintColor", tint_color)
        button.setProperty("iconColor", icon_color)
        button.setProperty("checkedBorderColor", checked_border_color)
        button.setStyleSheet("""
            QPushButton {
                background: %(card)s;
                border: 1.5px solid %(border)s;
                border-radius: 12px;
                text-align: left;
            }
            QPushButton:hover {
                background: %(tint_color)s;
                border-color: %(icon_color)s;
            }
            QPushButton:checked {
                background: %(tint_color)s;
                border: 2px solid %(icon_color)s;
            }
        """ % {**colors, "tint_color": tint_color, "icon_color": icon_color})

        card_layout = QHBoxLayout(button)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(9)

        icon_container = QLabel(button)
        icon_container.setFixedSize(34, 34)
        icon_container.setAlignment(Qt.AlignCenter)
        icon_container.setStyleSheet(
            "background: %s; border-radius: 10px; border: none;" % tint_color
        )
        icon_container.setPixmap(icon.icon(color=QColor(icon_color)).pixmap(QSize(18, 18)))
        icon_container.setAttribute(Qt.WA_TransparentForMouseEvents)
        card_layout.addWidget(icon_container, 0, Qt.AlignVCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        text_layout.setAlignment(Qt.AlignVCenter)
        title_label = QLabel(title, button)
        setFont(title_label, 12, QFont.DemiBold)
        title_label.setStyleSheet("color: %s; background: transparent;" % colors["primary"])
        title_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        description_label = QLabel(description, button)
        description_label.setWordWrap(False)
        setFont(description_label, 10)
        description_label.setStyleSheet("color: %s; background: transparent;" % colors["secondary"])
        description_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        text_layout.addWidget(title_label)
        text_layout.addWidget(description_label)
        card_layout.addLayout(text_layout, 1)

        group.addButton(button)
        mode_buttons[mode] = button
        cards_layout.addWidget(button, 1)

    layout.addWidget(cards_container)
    mode_buttons["image_edit"].setChecked(True)

    def _propagate_geometry_update():
        widget = parent
        for _ in range(12):
            if widget is None:
                break
            lyt = widget.layout()
            if lyt is not None:
                lyt.invalidate()
                lyt.activate()
            widget.updateGeometry()
            widget.adjustSize()
            parent_widget = widget.parentWidget()
            if parent_widget is None:
                break
            if isinstance(parent_widget, QAbstractScrollArea):
                parent_widget.updateGeometry()
                break
            widget = parent_widget

    def set_mode(mode: str):
        reference_card.setVisible(mode == "image_edit")
        _propagate_geometry_update()
        QTimer.singleShot(0, _propagate_geometry_update)

    mode_buttons["image_edit"].clicked.connect(lambda: set_mode("image_edit"))
    mode_buttons["text_to_image"].clicked.connect(lambda: set_mode("text_to_image"))
    return selector


class ResolutionSettingsCard(SettingsSection):
    CUSTOM_RATIO = "自定义"
    ASPECT_RATIOS = (
        ("1:1", 16, 16),
        ("4:3", 18, 14),
        ("3:4", 14, 18),
        ("3:2", 18, 12),
        ("2:3", 12, 18),
        ("16:9", 20, 11),
        ("9:16", 11, 20),
    )

    def __init__(self, params: LayeredTaskParams, parent=None):
        super().__init__(parent)
        self.params = params
        supported_ratios = {ratio for ratio, _, _ in self.ASPECT_RATIOS}
        selected_ratio = params.get_param("task.aspect_ratio")
        if selected_ratio in supported_ratios:
            self._selected_ratio = selected_ratio
            self._custom_ratio_width = 1
            self._custom_ratio_height = 1
        else:
            try:
                custom_width, custom_height = map(int, selected_ratio.split(":"))
            except (AttributeError, TypeError, ValueError):
                custom_width, custom_height = 0, 0
            if custom_width > 0 and custom_height > 0:
                self._selected_ratio = self.CUSTOM_RATIO
                self._custom_ratio_width = custom_width
                self._custom_ratio_height = custom_height
            else:
                self._selected_ratio = "1:1"
                self._custom_ratio_width = 1
                self._custom_ratio_height = 1
                self.params.set_param("task.aspect_ratio", self._selected_ratio)
        colors = _card_colors()
        self.content_layout.setSpacing(12)

        self.content_layout.addWidget(SectionHeading("⛶", "图像设置", parent=self))

        self.ratio_section = QWidget(self)
        ratio_section = QVBoxLayout(self.ratio_section)
        ratio_section.setContentsMargins(0, 0, 0, 0)
        ratio_section.setSpacing(6)
        ratio_label = QLabel("画面比例", self)
        setFont(ratio_label, 11)
        ratio_label.setStyleSheet("color: %s;" % colors["secondary"])
        ratio_section.addWidget(ratio_label)

        self._ratio_buttons = {}
        self._ratio_group = QButtonGroup(self)
        self._ratio_group.setExclusive(True)
        ratio_grid = QGridLayout()
        ratio_grid.setContentsMargins(0, 0, 0, 0)
        ratio_grid.setHorizontalSpacing(5)
        ratio_grid.setVerticalSpacing(5)
        ratio_options = (*self.ASPECT_RATIOS, (self.CUSTOM_RATIO, 16, 16))
        for index, (ratio, width, height) in enumerate(ratio_options):
            button = QPushButton(self)
            button.setCheckable(True)
            button.setMinimumHeight(54)
            button.setStyleSheet(self._option_button_style())
            button_layout = QVBoxLayout(button)
            button_layout.setContentsMargins(3, 5, 3, 5)
            button_layout.setSpacing(4)
            ratio_icon = QWidget(button)
            ratio_icon.setObjectName("ratioIcon")
            ratio_icon.setFixedSize(width, height)
            ratio_icon.setStyleSheet("""
                background: transparent;
                border: 2px solid %(secondary)s;
                border-radius: 2px;
            """ % colors)
            ratio_text = QLabel(ratio, button)
            setFont(ratio_text, 9, QFont.DemiBold)
            ratio_font = ratio_text.font()
            ratio_font.setStyleHint(QFont.Monospace)
            ratio_text.setFont(ratio_font)
            ratio_text.setStyleSheet("background: transparent;")
            button_layout.addWidget(ratio_icon, 0, Qt.AlignHCenter)
            button_layout.addWidget(ratio_text, 0, Qt.AlignHCenter)
            self._ratio_group.addButton(button)
            self._ratio_buttons[ratio] = button
            ratio_grid.addWidget(button, index // 4, index % 4)
            button.clicked.connect(lambda checked=False, selected=ratio: self._set_ratio(selected))
        ratio_section.addLayout(ratio_grid)

        self.custom_ratio_inputs = QWidget(self.ratio_section)
        custom_ratio_layout = QHBoxLayout(self.custom_ratio_inputs)
        custom_ratio_layout.setContentsMargins(0, 0, 0, 0)
        custom_ratio_layout.setSpacing(7)
        ratio_input_label = QLabel("宽高比", self.custom_ratio_inputs)
        setFont(ratio_input_label, 11)
        ratio_input_label.setStyleSheet("color: %s;" % colors["secondary"])
        custom_ratio_layout.addWidget(ratio_input_label)
        custom_ratio_layout.addStretch(1)
        self.ratio_width_input = self._ratio_input("宽")
        self.ratio_height_input = self._ratio_input("高")
        self.ratio_width_input.setText(str(self._custom_ratio_width))
        self.ratio_height_input.setText(str(self._custom_ratio_height))
        custom_ratio_layout.addWidget(self.ratio_width_input, 1)
        separator = QLabel(":", self.custom_ratio_inputs)
        separator.setAlignment(Qt.AlignCenter)
        separator.setStyleSheet("color: %s;" % colors["muted"])
        custom_ratio_layout.addWidget(separator)
        custom_ratio_layout.addWidget(self.ratio_height_input, 1)
        ratio_section.addWidget(self.custom_ratio_inputs)
        self.custom_ratio_hint = QLabel("例如 16:9", self.ratio_section)
        setFont(self.custom_ratio_hint, 9)
        self.custom_ratio_hint.setStyleSheet("color: %s;" % colors["muted"])
        ratio_section.addWidget(self.custom_ratio_hint)
        self.ratio_width_input.textChanged.connect(self._update_custom_ratio)
        self.ratio_height_input.textChanged.connect(self._update_custom_ratio)
        self.content_layout.addWidget(self.ratio_section)

        self._ratio_buttons[self._selected_ratio].setChecked(True)
        is_custom_ratio = self._selected_ratio == self.CUSTOM_RATIO
        self.custom_ratio_inputs.setVisible(is_custom_ratio)

    @staticmethod
    def _option_button_style():
        return """
            QPushButton {
                color: #475569;
                background: rgba(248, 250, 252, 0.85);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 10px;
            }
            QPushButton:hover {
                border-color: rgba(0, 103, 192, 0.32);
            }
            QPushButton:checked {
                color: #0067c0;
                background: #eff6ff;
                border: 1px solid #3b82f6;
            }
            QPushButton:checked QLabel {
                color: #0067c0;
            }
            QPushButton:checked QWidget#ratioIcon {
                border-color: #0067c0;
            }
        """

    def _ratio_input(self, placeholder: str) -> QLineEdit:
        input_field = QLineEdit(self)
        input_field.setObjectName("ratioInput")
        input_field.setPlaceholderText(placeholder)
        input_field.setAlignment(Qt.AlignCenter)
        input_field.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d*"), input_field))
        input_field.setFixedHeight(30)
        setFont(input_field, 12)
        input_field.setStyleSheet("""
            QLineEdit#ratioInput {
                color: #1e293b;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 0 10px;
                font-size: 12px;
                selection-background-color: #6366f1;
            }
            QLineEdit#ratioInput:hover {
                background: #fff;
                border-color: #cbd5e1;
            }
            QLineEdit#ratioInput:focus {
                background: #fff;
                border: 1px solid #818cf8;
            }
        """)
        return input_field

    def _set_ratio(self, ratio: str):
        self._selected_ratio = ratio
        is_custom = ratio == self.CUSTOM_RATIO
        self.custom_ratio_inputs.setVisible(is_custom)
        self.custom_ratio_hint.setVisible(is_custom)
        if not is_custom:
            self.params.set_param("task.aspect_ratio", ratio)
            return
        self._update_custom_ratio()

    def _update_custom_ratio(self):
        try:
            width = int(self.ratio_width_input.text())
            height = int(self.ratio_height_input.text())
        except ValueError:
            self.custom_ratio_hint.setText("请输入大于 0 的整数宽高比")
            self.custom_ratio_hint.setStyleSheet("color: #e11d48;")
            return
        if width <= 0 or height <= 0:
            self.custom_ratio_hint.setText("请输入大于 0 的整数宽高比")
            self.custom_ratio_hint.setStyleSheet("color: #e11d48;")
            return
        self._custom_ratio_width = width
        self._custom_ratio_height = height
        if self._selected_ratio == self.CUSTOM_RATIO:
            self.params.set_param("task.aspect_ratio", f"{width}:{height}")
        colors = _card_colors()
        self.custom_ratio_hint.setText("例如 16:9")
        self.custom_ratio_hint.setStyleSheet("color: %s;" % colors["muted"])


class GeneralEditSettingsPanel(QWidget):
    def __init__(self, params: LayeredTaskParams, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        self.reference_image_card = ReferenceImageCard(params, self)
        self.mode_selector = _create_edit_mode_selector(self, self.reference_image_card)
        layout.addWidget(self.mode_selector)
        layout.addWidget(PromptEditorCard(params, self))
        layout.addWidget(self.reference_image_card)
        layout.addWidget(ResolutionSettingsCard(params, self))


class PrecisionEditSettingsPanel(QWidget):
    def __init__(self, params: LayeredTaskParams, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        self.reference_image_card = ReferenceImageCard(params, self)
        self.mode_selector = _create_edit_mode_selector(self, self.reference_image_card)
        layout.addWidget(self.mode_selector)
        layout.addWidget(PromptEditorCard(params, self))
        layout.addWidget(self.reference_image_card)
        layout.addWidget(ResolutionSettingsCard(params, self))


class _CurrentPageStackedWidget(QStackedWidget):
    def sizeHint(self) -> QSize:
        size_hint = super().sizeHint()
        current_widget = self.currentWidget()
        if current_widget is not None:
            size_hint.setHeight(current_widget.sizeHint().height())
        return size_hint

    def minimumSizeHint(self) -> QSize:
        size_hint = super().minimumSizeHint()
        current_widget = self.currentWidget()
        if current_widget is not None:
            size_hint.setHeight(current_widget.minimumSizeHint().height())
        return size_hint


MODEL_SETTINGS_PANELS = {
    "general_edit": GeneralEditSettingsPanel,
    "precision_edit": PrecisionEditSettingsPanel,
}


class ParameterSettingsCard(QFrame):
    def __init__(self, params: LayeredTaskParams, parent=None):
        super().__init__(parent)
        self.setObjectName("parameterSettings")
        colors = _card_colors()
        self.setStyleSheet("""
            QFrame#parameterSettings {
                background-color: %(background)s;
                border: 1px solid %(border)s;
                border-radius: 8px;
            }
        """ % {"background": colors["card"], "border": colors["border"]})
        self._panels = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(10)

        header = QWidget(self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 4, 0, 0)
        header_layout.setSpacing(8)
        icon = QLabel("⚙️")
        setFont(icon, 13)
        icon.setStyleSheet("color: #555555; background: transparent;")
        title = QLabel(self.tr("参数设置"))
        setFont(title, 13, QFont.DemiBold)
        title.setStyleSheet("color: #555555; background: transparent;")
        divider = QFrame(header)
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color: #e2e8f0;")
        header_layout.addWidget(icon)
        header_layout.addWidget(title)
        header_layout.addWidget(divider, 1)
        layout.addWidget(header)

        self.stack = _CurrentPageStackedWidget(self)
        self.stack.setStyleSheet("QStackedWidget { background: transparent; border: none; }")
        layout.addWidget(self.stack)

        for model_name, panel_type in MODEL_SETTINGS_PANELS.items():
            panel = panel_type(params, self.stack)
            self._panels[model_name] = panel
            self.stack.addWidget(panel)

        self._placeholder = QLabel(self.tr("该模型暂无参数设置"), self)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(self._placeholder)

    def sizeHint(self) -> QSize:
        layout = self.layout()
        if layout is None:
            return super().sizeHint()
        frame_size = self.frameWidth() * 2
        return layout.sizeHint() + QSize(frame_size, frame_size)

    def minimumSizeHint(self) -> QSize:
        layout = self.layout()
        if layout is None:
            return super().minimumSizeHint()
        frame_size = self.frameWidth() * 2
        return layout.minimumSize() + QSize(frame_size, frame_size)

    def set_model(self, model_name: str):
        panel = self._panels.get(model_name, self._placeholder)
        for stacked_panel in [*self._panels.values(), self._placeholder]:
            if stacked_panel is not panel:
                stacked_panel.hide()
        if self.stack.currentWidget() is not panel:
            self.stack.setCurrentWidget(panel)
        panel.show()
        panel_layout = panel.layout()
        if panel_layout is not None:
            panel_layout.activate()
        self.stack.updateGeometry()
        self.updateGeometry()

        def refresh_geometry():
            if not self.isVisible():
                return
            self.layout().activate()
            self.stack.updateGeometry()
            self.update()

        QTimer.singleShot(0, refresh_geometry)