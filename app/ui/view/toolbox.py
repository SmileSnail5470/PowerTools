import json
import os
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget
from app.ui.library.qfluentwidgets import FluentIcon, MaskDialogBase, setFont
from app.ui.widgets.media_split_dialog import MediaSplitDialog, compose_split_outputs
from app.ui.widgets.image_preview_widget import ScrollBar


class ToolCard(QFrame):
    def __init__(self, icon: FluentIcon, eyebrow: str, title: str, description: str, details: list[str], action_text: str, on_click, accent: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ToolCard")
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        accent_end = "#35A7F2" if accent == "#0878C9" else "#A273F4"
        accent_soft = "#EFF7FE" if accent == "#0878C9" else "#F5F0FE"
        self.setStyleSheet("""
            QFrame#ToolCard {
                background: #FFFFFF;
                border: 1px solid #E3E9F2;
                border-radius: 19px;
            }
            QFrame#ToolCard:hover {
                border: 1px solid #C8D9EE;
                background: #FCFDFF;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(10)

        accent_line = QFrame()
        accent_line.setFixedHeight(4)
        accent_line.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent}, stop:1 {accent_end}); border: none; border-radius: 2px;")
        layout.addWidget(accent_line)

        header = QHBoxLayout()
        icon_tile = QLabel()
        icon_tile.setFixedSize(50, 50)
        icon_tile.setAlignment(Qt.AlignCenter)
        icon_tile.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 {accent_soft}); border: 1px solid {accent_soft}; border-radius: 15px;")
        icon_tile.setPixmap(icon.icon().pixmap(QSize(26, 26)))
        header.addWidget(icon_tile)

        heading = QVBoxLayout()
        heading.setSpacing(5)
        eyebrow_label = QLabel(eyebrow)
        eyebrow_label.setAlignment(Qt.AlignCenter)
        eyebrow_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        eyebrow_label.setStyleSheet(f"""
            QLabel {{
                color: {accent};
                background: {accent_soft};
                border: 1px solid {accent_soft};
                border-radius: 9px;
                padding: 3px 9px;
                font-size: 10px;
                font-weight: 700;
            }}
        """)
        title_label = QLabel(title)
        setFont(title_label, 18, QFont.DemiBold)
        title_label.setStyleSheet("color: #17243A;")
        heading.addWidget(eyebrow_label)
        heading.addWidget(title_label)
        header.addLayout(heading)
        header.addStretch()
        layout.addLayout(header)

        description_label = QLabel(description)
        description_label.setObjectName("toolDescription")
        description_label.setWordWrap(True)
        description_label.setMinimumHeight(42)
        description_label.setStyleSheet("""
            QLabel#toolDescription {
                color: #687386;
                background: transparent;
                border: none;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        layout.addWidget(description_label)

        details_row = QHBoxLayout()
        details_row.setSpacing(7)
        for detail in details:
            detail_label = QLabel(detail)
            detail_label.setAlignment(Qt.AlignCenter)
            detail_label.setStyleSheet(f"""
                QLabel {{
                    color: #536176;
                    background: {accent_soft};
                    border: 1px solid #E9EEF5;
                    border-radius: 12px;
                    padding: 5px 11px;
                    font-size: 11px;
                }}
            """)
            details_row.addWidget(detail_label)
        details_row.addStretch()
        layout.addLayout(details_row)
        layout.addStretch()

        action_button = QPushButton(action_text)
        action_button.setCursor(Qt.PointingHandCursor)
        action_button.setFixedHeight(42)
        action_button.setStyleSheet(f"""
            QPushButton {{
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent}, stop:1 {accent_end});
                border: none;
                border-radius: 13px;
                padding: 0 17px;
                text-align: center;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent_end}, stop:1 {accent});
            }}
            QPushButton:pressed {{ background: #1769AA; }}
        """)
        action_button.clicked.connect(on_click)
        layout.addWidget(action_button)


class ComposeDialog(MaskDialogBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("合成工具")
        self.widget.setFixedSize(780, 600)
        self.widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._hBoxLayout.setAlignment(self.widget, Qt.AlignCenter)
        self.widget.setAttribute(Qt.WA_StyledBackground, True)
        self.widget.setStyleSheet("QFrame#centerWidget { background: #FFFFFF; border-radius: 14px; }")
        self.setShadowEffect(60, (0, 12), QColor(15, 23, 42, 60))
        self.setMaskColor(QColor(15, 23, 42, 100))
        self._metadata_path = ""
        self._regions: list[dict] = []
        self._processed_paths: list[str] = []
        self._region_rows: list[tuple[QLabel, QPushButton]] = []

        root = QVBoxLayout(self.widget)
        root.setContentsMargins(12, 8, 12, 12)
        root.setSpacing(8)
        root.addWidget(self._build_header())

        body = QFrame()
        body.setObjectName("composeBody")
        body.setStyleSheet("""
            QFrame#composeBody {
                background: #F7F9FC;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
            }
        """)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 14, 16, 14)
        body_layout.setSpacing(12)
        root.addWidget(body, 1)

        subtitle = QLabel("选择已裁剪目录，为每个裁剪区域指定算法处理后的图片或视频。")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #687386; font-size: 13px;")
        body_layout.addWidget(subtitle)

        project_row = QHBoxLayout()
        self._project_label = QLabel("尚未选择已裁剪目录")
        self._project_label.setMinimumWidth(0)
        self._project_label.setStyleSheet("color: #687386; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 9px; padding: 9px 12px;")
        self._project_label.setToolTip("")
        project_button = QPushButton("选择裁剪目录…")
        project_button.setCursor(Qt.PointingHandCursor)
        project_button.setFixedHeight(38)
        project_button.setStyleSheet("""
            QPushButton {
                color: #FFFFFF;
                background: #0078D4;
                border: none;
                border-radius: 9px;
                padding: 0 15px;
                font-weight: 600;
            }
            QPushButton:hover { background: #106EBE; }
        """)
        project_button.clicked.connect(self._select_project)
        project_row.addWidget(self._project_label, 1)
        project_row.addWidget(project_button)
        body_layout.addLayout(project_row)

        self._region_container = QWidget()
        self._region_container.setStyleSheet("background: transparent;")
        self._region_layout = QVBoxLayout(self._region_container)
        self._region_layout.setContentsMargins(0, 0, 0, 0)
        self._region_layout.setSpacing(7)
        self._region_scroll = QScrollArea()
        self._v_scroll = ScrollBar(Qt.Vertical,  self._region_scroll)
        self._h_scroll = ScrollBar(Qt.Horizontal,  self._region_scroll)
        self._region_scroll.setVerticalScrollBar(self._v_scroll)
        self._region_scroll.setHorizontalScrollBar(self._h_scroll)
        self._region_scroll.setWidgetResizable(True)
        self._region_scroll.setFrameShape(QFrame.NoFrame)
        self._region_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._region_scroll.setWidget(self._region_container)
        body_layout.addWidget(self._region_scroll)
        body_layout.addStretch(1)

        footer = QHBoxLayout()
        self._status_label = QLabel("每个裁剪区域都选择处理后的文件后即可开始合成。")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #7A8494; font-size: 12px;")
        footer.addWidget(self._status_label, 1)
        self._compose_button = QPushButton("合成并保存")
        self._compose_button.setCursor(Qt.PointingHandCursor)
        self._compose_button.setEnabled(False)
        self._compose_button.setFixedHeight(42)
        self._compose_button.setStyleSheet("""
            QPushButton {
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0078D4, stop:1 #38A3E8);
                border: none;
                border-radius: 10px;
                padding: 0 18px;
                font-weight: 600;
            }
            QPushButton:disabled { color: #A7B0BE; background: #E8ECF2; }
            QPushButton:hover:!disabled { background: #106EBE; }
        """)
        self._compose_button.clicked.connect(self._compose)
        footer.addWidget(self._compose_button)
        body_layout.addLayout(footer)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(60)
        header.setStyleSheet("background: #FFFFFF; border: none;")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(10)

        badge = QLabel("▦")
        badge.setFixedSize(32, 32)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet("""
            QLabel {
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0078D4, stop:1 #38BDF8);
                border: none;
                border-radius: 10px;
                font-size: 16px;
            }
        """)
        layout.addWidget(badge)

        title_block = QVBoxLayout()
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(2)
        title = QLabel("图片/视频结果合成")
        title.setStyleSheet("font-size: 14px; font-weight: 600; color: #0F172A;")
        caption = QLabel("选择处理结果并还原完整画面")
        caption.setStyleSheet("font-size: 11px; color: #64748B;")
        title_block.addWidget(title)
        title_block.addWidget(caption)
        layout.addLayout(title_block)
        layout.addStretch()

        close_button = QPushButton("×")
        close_button.setFixedSize(30, 30)
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.setStyleSheet("""
            QPushButton {
                color: #64748B;
                background: #F1F5F9;
                border: 1px solid #E2E8F0;
                border-radius: 15px;
                font-size: 15px;
            }
            QPushButton:hover {
                color: #B91C1C;
                background: #FEE2E2;
                border-color: #FECACA;
            }
        """)
        close_button.clicked.connect(self.reject)
        layout.addWidget(close_button)
        return header

    def _select_project(self):
        metadata_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择裁剪项目",
            "",
            "裁剪项目 (crop_metadata.json);;JSON 文件 (*.json)",
        )
        if not metadata_path:
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            if not isinstance(metadata, dict):
                raise ValueError("裁剪项目格式无效。")
            regions = metadata.get("regions")
            if not isinstance(regions, list) or not regions:
                raise ValueError("项目中没有可合成的裁剪区域。")
            if not all(isinstance(region, dict) for region in regions):
                raise ValueError("裁剪项目中的区域信息格式无效。")
        except (OSError, json.JSONDecodeError, ValueError, AttributeError) as exc:
            QMessageBox.warning(self, "无法读取裁剪项目", str(exc))
            return
        self._metadata_path = metadata_path
        self._regions = regions
        self._processed_paths = [""] * len(regions)
        self._project_label.setText(os.path.basename(metadata_path))
        self._project_label.setToolTip(metadata_path)
        self._status_label.setText(f"项目包含 {len(regions)} 个区域，请逐项选择处理后的文件。")
        self._rebuild_region_rows()
        self._update_compose_button()

    def _rebuild_region_rows(self):
        self._region_container.setMinimumHeight(0)
        while self._region_layout.count():
            item = self._region_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._region_rows.clear()

        def file_selection_callback(region_index: int):
            return lambda: self._select_processed_file(region_index)

        for index, region in enumerate(self._regions):
            row = QFrame()
            row.setStyleSheet("QFrame { background: #FFFFFF; border: 1px solid #E7EBF1; border-radius: 9px; }")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(12, 8, 10, 8)
            row_layout.setSpacing(10)

            info = QLabel(f"区域 {index + 1:02d}  ·  {region.get('width', '?')} × {region.get('height', '?')}")
            info.setStyleSheet("color: #344054; font-size: 12px;")
            path_label = QLabel("请选择裁剪区域算法处理后的文件")
            path_label.setMinimumWidth(0)
            path_label.setStyleSheet("color: #8993A2; font-size: 11px;")
            path_label.setToolTip("")
            choose_button = QPushButton("选择文件")
            choose_button.setCursor(Qt.PointingHandCursor)
            choose_button.setStyleSheet(
                "QPushButton { color: #0067B8; background: #EFF6FC; border: none; "
                "border-radius: 7px; padding: 7px 11px; }"
                "QPushButton:hover { background: #E2F0FC; }")
            choose_button.clicked.connect(file_selection_callback(index))
            row_layout.addWidget(info)
            row_layout.addWidget(path_label, 1)
            row_layout.addWidget(choose_button)
            self._region_layout.addWidget(row)
            self._region_rows.append((path_label, choose_button))

        QTimer.singleShot(0, self._fit_region_scroll)

    def _fit_region_scroll(self):
        self._region_layout.activate()
        content_height = self._region_layout.sizeHint().height()
        self._region_container.setMinimumHeight(content_height)
        self._region_scroll.setFixedHeight(min(260, content_height))

    def _select_processed_file(self, index: int):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择区域 {index + 1} 的处理结果",
            "",
            "图片或视频 (*.png *.jpg *.jpeg *.bmp *.webp *.avif *.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;"
            "所有文件 (*)",
        )
        if not file_path:
            return
        self._processed_paths[index] = file_path
        path_label = self._region_rows[index][0]
        path_label.setText(os.path.basename(file_path))
        path_label.setToolTip(file_path)
        path_label.setStyleSheet("color: #344054; font-size: 11px;")
        self._update_compose_button()

    def _update_compose_button(self):
        ready = bool(self._regions) and all(self._processed_paths)
        self._compose_button.setEnabled(ready)

    def _compose(self):
        if not self._metadata_path or not all(self._processed_paths):
            return
        try:
            with open(self._metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "无法读取裁剪项目", str(exc))
            return
        if not isinstance(metadata, dict):
            QMessageBox.warning(self, "无法读取裁剪项目", "裁剪项目格式无效。")
            return
        source_path = metadata.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            source_path = "composed"
        stem = os.path.splitext(os.path.basename(source_path))[0]
        extension = os.path.splitext(source_path)[1] if metadata.get("is_video") else ".png"
        if not extension:
            extension = ".mp4"
        export_dir = metadata.get("export_dir")
        if not isinstance(export_dir, str) or not export_dir:
            export_dir = os.path.dirname(self._metadata_path)
        default_path = os.path.join(export_dir, f"{stem}_composed{extension}")
        output_path, _ = QFileDialog.getSaveFileName(self, "保存合成结果", default_path, "所有文件 (*)")
        if not output_path:
            return
        self._compose_button.setEnabled(False)
        self._compose_button.setText("正在合成…")
        self.setCursor(Qt.WaitCursor)
        try:
            final_path = compose_split_outputs(self._metadata_path, self._processed_paths, output_path,)
        except Exception as exc:
            QMessageBox.critical(self, "合成失败", str(exc))
            self._compose_button.setText("合成并保存")
            self.unsetCursor()
            self._update_compose_button()
            return
        self.unsetCursor()
        self._status_label.setText(f"已保存：{final_path}")
        self._compose_button.setText("合成完成")
        QMessageBox.information(self, "合成完成", f"合成结果已保存至：\n{final_path}")


class Toolbox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Toolbox")
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea(self)
        _v_scroll = ScrollBar(Qt.Vertical, scroll)
        _h_scroll = ScrollBar(Qt.Horizontal, scroll)
        scroll.setVerticalScrollBar(_v_scroll)
        scroll.setHorizontalScrollBar(_h_scroll)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: #F5F7FB; border: none; }")
        root.addWidget(scroll)

        content = QWidget()
        content.setStyleSheet("background: #F5F7FB;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 18, 16, 22)
        layout.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("ToolboxHero")
        hero.setMinimumHeight(168)
        hero.setStyleSheet("""
            QFrame#ToolboxHero {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4A7AA6, stop:0.52 #6679AE, stop:1 #8875A2);
                border: 1px solid #7180A8;
                border-radius: 22px;
            }
        """)
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 18, 24, 18)
        hero_layout.setSpacing(6)
        eyebrow = QLabel("POWERTOOLS  ·  WORKSPACE")
        eyebrow.setAlignment(Qt.AlignCenter)
        eyebrow.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        eyebrow.setStyleSheet("""
            QLabel {
                color: white;
                background: rgba(255, 255, 255, 0.16);
                border: 1px solid rgba(255, 255, 255, 0.28);
                border-radius: 10px;
                padding: 5px 11px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }
        """)
        title = QLabel("常用工具箱")
        title.setAlignment(Qt.AlignCenter)
        setFont(title, 29, QFont.DemiBold)
        title.setStyleSheet("color: #FFFFFF; background: transparent;")
        subtitle = QLabel("轻松完成媒体裁剪与结果合成，更多实用工具即将加入。")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.84); background: transparent; font-size: 13px;")
        hero_layout.addWidget(eyebrow, 0, Qt.AlignHCenter)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        layout.addWidget(hero)

        section_header = QHBoxLayout()
        section_title = QLabel("媒体工具")
        setFont(section_title, 17, QFont.DemiBold)
        section_title.setStyleSheet("color: #202B3C;")
        section_hint = QLabel("精选轻量工具")
        section_hint.setAlignment(Qt.AlignCenter)
        section_hint.setStyleSheet("""
            QLabel {
                color: #748198;
                background: #FFFFFF;
                border: 1px solid #E5EAF1;
                border-radius: 10px;
                padding: 4px 10px;
                font-size: 11px;
            }
        """)
        section_header.addWidget(section_title)
        section_header.addSpacing(10)
        section_header.addWidget(section_hint)
        section_header.addStretch()
        layout.addLayout(section_header)

        cards = QGridLayout()
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setHorizontalSpacing(12)
        cards.setVerticalSpacing(12)
        cards.addWidget(
            ToolCard(
                FluentIcon.CUT,
                "图片 · 视频",
                "媒体裁剪",
                "按画面比例智能建议分区，也可以自定义裁剪区域并导出各区域文件。",
                ["智能分区", "自由调整", "批量导出"],
                "选择文件并开始裁剪   →",
                self._open_split,
                "#0878C9",
                content,
            ),
            0,
            0,
        )
        cards.addWidget(
            ToolCard(
                FluentIcon.PHOTO,
                "裁剪工作流",
                "结果合成",
                "导入裁剪项目，为每个区域选择处理后的文件，再合成为完整图片或视频。",
                ["读取项目元数据", "区域对应", "原位合成"],
                "选择项目并开始合成   →",
                self._open_compose,
                "#6658C8",
                content,
            ),
            0,
            1,
        )
        layout.addLayout(cards)

        layout.addStretch()
        scroll.setWidget(content)

    def _open_split(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片或视频",
            "",
            "图片或视频 (*.png *.jpg *.jpeg *.bmp *.webp *.avif *.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;"
            "所有文件 (*)",
        )
        if file_path:
            dialog = MediaSplitDialog(file_path, self.window())
            dialog.exec()

    def _open_compose(self):
        dialog = ComposeDialog(self.window())
        dialog.exec()
