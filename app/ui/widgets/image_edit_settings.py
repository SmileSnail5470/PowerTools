import os
import sys
from PySide6.QtCore import QRegularExpression, Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QFileDialog, QButtonGroup, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QStackedWidget, QTextEdit, QVBoxLayout, QWidget
)
from app.ui.common.event_bus import global_event_bus
from app.ui.common.task_params import LayeredTaskParams
from app.ui.library.qfluentwidgets import setFont
from app.ui.widgets.watermark_interactive_widget import AreaSelectorDialog


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
        if valid_paths:
            self._set_files([*self._file_paths, *valid_paths])
        if invalid_paths:
            self.drop_zone.set_status(self.tr("有 {0} 个文件无法读取，已忽略").format(len(invalid_paths)), is_error=True,)
        else:
            self.drop_zone.set_status(self.tr("支持多选或拖拽多张参考图片"))

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
            if index == 0:
                badge = QLabel("主图", tile)
                badge.setAlignment(Qt.AlignCenter)
                badge.setStyleSheet("color: %s; background: %s; border-radius: 5px; padding: 2px;" % (colors["accent"], colors["tint"]))
                tile_layout.addWidget(badge)
            thumbnail = QLabel(tile)
            thumbnail.setFixedSize(72, 60)
            thumbnail.setAlignment(Qt.AlignCenter)
            thumbnail.setPixmap(QPixmap(path).scaled(72, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            tile_layout.addWidget(thumbnail, 0, Qt.AlignHCenter)
            filename = QLabel(os.path.basename(path), tile)
            filename.setFixedWidth(84)
            filename.setAlignment(Qt.AlignCenter)
            filename.setToolTip(path)
            filename.setStyleSheet("color: %s;" % colors["primary"])
            setFont(filename, 8)
            tile_layout.addWidget(filename)
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
        dialog = AreaSelectorDialog(file_path=self._file_paths[0], single_area_only=False, parent=self.window())
        dialog.exec()
        boxes = dialog.get_results()
        self.params.set_param(self.mask_path, boxes)
        if boxes:
            self.area_info.setText(self.tr("{0} 个区域已选择").format(len(boxes)))
        else:
            self.area_info.setText(self.tr("尚未选择重绘区域"))


class ResolutionSettingsCard(SettingsSection):
    RESOLUTIONS = {
        "1K": {
            "1:1": (1024, 1024),
            "4:3": (1152, 864),
            "3:4": (864, 1152),
            "3:2": (1200, 800),
            "2:3": (800, 1200),
            "16:9": (1280, 720),
            "9:16": (720, 1280),
        },
        "1.5K": {
            "1:1": (1536, 1536),
            "4:3": (1728, 1296),
            "3:4": (1296, 1728),
            "3:2": (1800, 1200),
            "2:3": (1200, 1800),
            "16:9": (1920, 1080),
            "9:16": (1080, 1920),
        },
        "2K": {
            "1:1": (2048, 2048),
            "4:3": (2304, 1728),
            "3:4": (1728, 2304),
            "3:2": (2400, 1600),
            "2:3": (1600, 2400),
            "16:9": (2560, 1440),
            "9:16": (1440, 2560),
        },
    }
    SIZE_TIERS = (
        ("1K", "标准高清"),
        ("1.5K", "细腻增强"),
        ("2K", "超清原画"),
        ("自定义", "输入分辨率"),
    )
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
        self._selected_tier = params.get_param("task.resolution_tier")
        supported_ratios = {ratio for ratio, _, _ in self.ASPECT_RATIOS}
        self._selected_ratio = params.get_param("task.aspect_ratio")
        if self._selected_ratio not in supported_ratios:
            self._selected_ratio = "1:1"
            self.params.set_param("task.aspect_ratio", self._selected_ratio)
        colors = _card_colors()
        self.content_layout.setSpacing(12)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(SectionHeading("⛶", "分辨率设置", parent=self), 1)
        self.exact_resolution = QLabel(self)
        setFont(self.exact_resolution, 10, QFont.DemiBold)
        exact_font = self.exact_resolution.font()
        exact_font.setStyleHint(QFont.Monospace)
        self.exact_resolution.setFont(exact_font)
        self.exact_resolution.setStyleSheet("""
            QLabel {
                color: %(accent)s;
                background: %(tint)s;
                border: 1px solid rgba(59, 130, 246, 0.15);
                border-radius: 10px;
                padding: 3px 8px;
            }
        """ % colors)
        header.addWidget(self.exact_resolution, 0, Qt.AlignVCenter)
        self.content_layout.addLayout(header)

        size_section = QVBoxLayout()
        size_section.setContentsMargins(0, 0, 0, 0)
        size_section.setSpacing(6)
        size_label = QLabel("图像大小", self)
        setFont(size_label, 10)
        size_label.setStyleSheet("color: %s;" % colors["secondary"])
        size_section.addWidget(size_label)

        self._tier_buttons = {}
        self._tier_group = QButtonGroup(self)
        self._tier_group.setExclusive(True)
        tier_grid = QGridLayout()
        tier_grid.setContentsMargins(0, 0, 0, 0)
        tier_grid.setHorizontalSpacing(6)
        tier_grid.setVerticalSpacing(6)
        for index, (tier, description) in enumerate(self.SIZE_TIERS):
            button = QPushButton(self)
            button.setCheckable(True)
            button.setMinimumHeight(52)
            button.setStyleSheet(self._option_button_style())
            button_layout = QVBoxLayout(button)
            button_layout.setContentsMargins(4, 5, 4, 5)
            button_layout.setSpacing(2)
            tier_label = QLabel(tier, button)
            setFont(tier_label, 11, QFont.Bold)
            tier_font = tier_label.font()
            tier_font.setStyleHint(QFont.Monospace)
            tier_label.setFont(tier_font)
            tier_label.setStyleSheet("background: transparent;")
            description_label = QLabel(description, button)
            setFont(description_label, 8)
            description_label.setStyleSheet("background: transparent;")
            button_layout.addWidget(tier_label, 0, Qt.AlignHCenter)
            button_layout.addWidget(description_label, 0, Qt.AlignHCenter)
            button_layout.setAlignment(Qt.AlignCenter)
            self._tier_group.addButton(button)
            self._tier_buttons[tier] = button
            tier_grid.addWidget(button, 0, index)
            button.clicked.connect(lambda checked=False, selected=tier: self._set_tier(selected))
        size_section.addLayout(tier_grid)

        self.custom_dimensions = QWidget(self)
        dimensions_layout = QHBoxLayout(self.custom_dimensions)
        dimensions_layout.setContentsMargins(0, 0, 0, 0)
        dimensions_layout.setSpacing(7)
        self.width_input = self._dimension_input("宽")
        self.height_input = self._dimension_input("高")
        dimensions_layout.addWidget(self.width_input, 1)
        separator = QLabel("×", self.custom_dimensions)
        separator.setAlignment(Qt.AlignCenter)
        separator.setStyleSheet("color: %s;" % colors["muted"])
        dimensions_layout.addWidget(separator)
        dimensions_layout.addWidget(self.height_input, 1)
        size_section.addWidget(self.custom_dimensions)
        self.custom_hint = QLabel("自定义尺寸不随画面比例调整", self)
        setFont(self.custom_hint, 9)
        self.custom_hint.setStyleSheet("color: %s;" % colors["muted"])
        size_section.addWidget(self.custom_hint)
        self.content_layout.addLayout(size_section)

        self.ratio_section = QWidget(self)
        ratio_section = QVBoxLayout(self.ratio_section)
        ratio_section.setContentsMargins(0, 0, 0, 0)
        ratio_section.setSpacing(6)
        ratio_label = QLabel("画面比例", self)
        setFont(ratio_label, 10)
        ratio_label.setStyleSheet("color: %s;" % colors["secondary"])
        ratio_section.addWidget(ratio_label)

        self._ratio_buttons = {}
        self._ratio_group = QButtonGroup(self)
        self._ratio_group.setExclusive(True)
        ratio_grid = QGridLayout()
        ratio_grid.setContentsMargins(0, 0, 0, 0)
        ratio_grid.setHorizontalSpacing(5)
        ratio_grid.setVerticalSpacing(5)
        for index, (ratio, width, height) in enumerate(self.ASPECT_RATIOS):
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
        self.content_layout.addWidget(self.ratio_section)

        self._tier_buttons.get(self._selected_tier, self._tier_buttons["1K"]).setChecked(True)
        self._ratio_buttons.get(self._selected_ratio, self._ratio_buttons["1:1"]).setChecked(True)
        self.width_input.setText(str(params.get_param("task.custom_width")))
        self.height_input.setText(str(params.get_param("task.custom_height")))
        self.custom_dimensions.setVisible(self._selected_tier == "自定义")
        self.custom_hint.setVisible(self._selected_tier == "自定义")
        self.ratio_section.setVisible(self._selected_tier != "自定义")
        self.width_input.textChanged.connect(self._update_custom_resolution)
        self.height_input.textChanged.connect(self._update_custom_resolution)
        self._update_exact_resolution()

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

    def _dimension_input(self, placeholder: str) -> QLineEdit:
        colors = _card_colors()
        input_field = QLineEdit(self)
        input_field.setPlaceholderText(placeholder)
        input_field.setAlignment(Qt.AlignCenter)
        input_field.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d*"), input_field))
        setFont(input_field, 10)
        input_field.setStyleSheet("""
            QLineEdit {
                color: %(primary)s;
                background: %(sub)s;
                border: 1px solid %(border)s;
                border-radius: 8px;
                padding: 6px 8px;
            }
            QLineEdit:focus {
                border: 1px solid %(accent)s;
            }
        """ % colors)
        return input_field

    def _set_tier(self, tier: str):
        self._selected_tier = tier
        self.params.set_param("task.resolution_tier", tier)
        self.custom_dimensions.setVisible(tier == "自定义")
        self.custom_hint.setVisible(tier == "自定义")
        self.ratio_section.setVisible(tier != "自定义")
        self._update_exact_resolution()

    def _set_ratio(self, ratio: str):
        self._selected_ratio = ratio
        self.params.set_param("task.aspect_ratio", ratio)
        self._update_exact_resolution()

    def _update_custom_resolution(self):
        try:
            width = int(self.width_input.text())
            height = int(self.height_input.text())
        except ValueError:
            self._show_invalid_resolution()
            return
        if width <= 0 or height <= 0:
            self._show_invalid_resolution()
            return
        self.params.set_param("task.custom_width", width)
        self.params.set_param("task.custom_height", height)
        self._update_exact_resolution(width, height)

    def _show_invalid_resolution(self):
        self.exact_resolution.setText("请输入有效分辨率")
        self.exact_resolution.setStyleSheet("""
            QLabel {
                color: #e11d48;
                background: #fff1f2;
                border: 1px solid rgba(225, 29, 72, 0.2);
                border-radius: 10px;
                padding: 3px 8px;
            }
        """)

    def _update_exact_resolution(self, custom_width=None, custom_height=None):
        if self._selected_tier == "自定义":
            if custom_width is None or custom_height is None:
                try:
                    custom_width = int(self.width_input.text())
                    custom_height = int(self.height_input.text())
                except ValueError:
                    self._show_invalid_resolution()
                    return
            if custom_width <= 0 or custom_height <= 0:
                self._show_invalid_resolution()
                return
            width, height = custom_width, custom_height
            exact_resolution_info = f"自定义 → {width} × {height} px"
        else:
            width, height = self.RESOLUTIONS[self._selected_tier][self._selected_ratio]
            exact_resolution_info = f"{self._selected_tier} · {self._selected_ratio} → {width} × {height} px"

        colors = _card_colors()
        self.exact_resolution.setText(exact_resolution_info)
        self.exact_resolution.setStyleSheet("""
            QLabel {
                color: %(accent)s;
                background: %(tint)s;
                border: 1px solid rgba(59, 130, 246, 0.15);
                border-radius: 10px;
                padding: 3px 8px;
            }
        """ % colors)


class GeneralEditSettingsPanel(QWidget):
    def __init__(self, params: LayeredTaskParams, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(PromptEditorCard(params, self))
        layout.addWidget(ReferenceImageCard(params, self))
        layout.addWidget(ResolutionSettingsCard(params, self))


class PrecisionEditSettingsPanel(QWidget):
    def __init__(self, params: LayeredTaskParams, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(PromptEditorCard(params, self))
        layout.addWidget(ReferenceImageCard(params, self))
        layout.addWidget(ResolutionSettingsCard(params, self))


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

        self.stack = QStackedWidget(self)
        self.stack.setStyleSheet("QStackedWidget { background: transparent; border: none; }")
        layout.addWidget(self.stack)

        for model_name, panel_type in MODEL_SETTINGS_PANELS.items():
            panel = panel_type(params, self.stack)
            self._panels[model_name] = panel
            self.stack.addWidget(panel)

        self._placeholder = QLabel(self.tr("该模型暂无参数设置"), self)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(self._placeholder)

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