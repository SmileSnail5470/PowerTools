"""GUI日志查看器组件"""

from PySide6.QtCore import Qt, QTimer, Signal, QMutex, QMutexLocker
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QComboBox, QLabel, QCheckBox, QFileDialog, QMenu, QInputDialog
)
from PySide6.QtGui import QTextCharFormat, QColor, QTextCursor, QFont, QAction
from app.ui.library.qfluentwidgets import (
    setFont, PrimaryPushButton, PushButton, LineEdit,
    ComboBox, CheckBox, FluentIcon, InfoBar, InfoBarPosition
)
from app.utils.logger.handlers import QtLogHandler
from app.utils.logger import get_log_manager
import os
from datetime import datetime


class LogViewerWidget(QWidget):
    """日志查看器组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.log_handler: QtLogHandler = None
        self.log_buffer = []
        self.max_log_lines = 10000  # 最大显示行数
        self.current_filter_level = "ALL"
        self.filter_keyword = ""
        self.mutex = QMutex()
        
        self.setup_ui()
        self.connect_log_handler()
        
        # 自动刷新定时器
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_logs)
        self.refresh_timer.start(100)  # 每100ms刷新一次
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # 工具栏
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)
        
        # 日志显示区域
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        setFont(self.log_text, 10)
        
        # 设置样式
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        layout.addWidget(self.log_text)
        
        # 状态栏
        status_bar = self.create_status_bar()
        layout.addWidget(status_bar)
    
    def create_toolbar(self) -> QWidget:
        """创建工具栏"""
        toolbar = QWidget(self)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(10)
        
        # 日志级别过滤
        level_label = QLabel(self.tr("级别:"))
        setFont(level_label, 12)
        toolbar_layout.addWidget(level_label)
        
        self.level_combo = ComboBox(self)
        self.level_combo.addItems([
            self.tr("全部"),
            self.tr("调试"),
            self.tr("信息"),
            self.tr("警告"),
            self.tr("错误"),
            self.tr("严重")
        ])
        self.level_combo.setCurrentIndex(0)
        self.level_combo.currentIndexChanged.connect(self.on_level_filter_changed)
        toolbar_layout.addWidget(self.level_combo)
        
        # 关键字搜索
        search_label = QLabel(self.tr("搜索:"))
        setFont(search_label, 12)
        toolbar_layout.addWidget(search_label)
        
        self.search_edit = LineEdit(self)
        self.search_edit.setPlaceholderText(self.tr("输入关键字..."))
        self.search_edit.textChanged.connect(self.on_search_changed)
        toolbar_layout.addWidget(self.search_edit)
        
        toolbar_layout.addStretch()
        
        # 清空按钮
        self.clear_btn = PushButton(self.tr("清空"), self)
        self.clear_btn.clicked.connect(self.clear_logs)
        toolbar_layout.addWidget(self.clear_btn)
        
        # 保存按钮
        self.save_btn = PushButton(self.tr("保存"), self)
        self.save_btn.clicked.connect(self.save_logs)
        toolbar_layout.addWidget(self.save_btn)
        
        # 自动滚动开关
        self.auto_scroll_check = CheckBox(self.tr("自动滚动"), self)
        self.auto_scroll_check.setChecked(True)
        toolbar_layout.addWidget(self.auto_scroll_check)
        
        # 字体大小
        font_label = QLabel(self.tr("字体大小:"))
        setFont(font_label, 12)
        toolbar_layout.addWidget(font_label)
        
        self.font_size_combo = ComboBox(self)
        self.font_size_combo.addItems(["8", "9", "10", "11", "12", "14", "16", "18"])
        self.font_size_combo.setCurrentText("10")
        self.font_size_combo.currentTextChanged.connect(self.on_font_size_changed)
        toolbar_layout.addWidget(self.font_size_combo)
        
        return toolbar
    
    def create_status_bar(self) -> QWidget:
        """创建状态栏"""
        status_bar = QWidget(self)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(5, 5, 5, 5)
        
        self.status_label = QLabel(self.tr("就绪"))
        setFont(self.status_label, 10)
        self.status_label.setStyleSheet("color: #888;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # 统计信息
        self.stats_label = QLabel("")
        setFont(self.stats_label, 10)
        self.stats_label.setStyleSheet("color: #888;")
        status_layout.addWidget(self.stats_label)
        
        return status_bar
    
    def connect_log_handler(self):
        """连接到日志处理器"""
        try:
            log_manager = get_log_manager()
            self.log_handler = log_manager.get_qt_handler()
            
            if self.log_handler:
                self.log_handler.log_message.connect(self.append_log)
                # 加载已缓冲的日志
                self.load_buffered_logs()
        except RuntimeError:
            # 日志系统未初始化，稍后重试
            pass
    
    def load_buffered_logs(self):
        """加载已缓冲的日志"""
        if self.log_handler:
            buffered_logs = self.log_handler.get_buffered_logs()
            for message, level, formatted_msg in buffered_logs:
                self.append_log(message, level, formatted_msg)
    
    def append_log(self, message: str, level: str, formatted_message: str):
        """添加日志到显示区域"""
        with QMutexLocker(self.mutex):
            # 添加到缓冲区
            self.log_buffer.append({
                'message': message,
                'level': level,
                'formatted': formatted_message,
                'timestamp': datetime.now()
            })
            
            # 限制缓冲区大小
            if len(self.log_buffer) > self.max_log_lines:
                self.log_buffer.pop(0)
    
    def refresh_logs(self):
        """刷新日志显示（定期调用）"""
        with QMutexLocker(self.mutex):
            if not self.log_buffer:
                return
            
            # 获取需要显示的日志
            new_logs = [log for log in self.log_buffer if self.should_display(log)]
            
            if not new_logs:
                return
            
            # 从缓冲区移除已处理的日志
            processed_ids = {id(log) for log in new_logs}
            self.log_buffer = [log for log in self.log_buffer if id(log) not in processed_ids]
            
            # 检查是否需要清理旧日志
            current_line_count = self.log_text.document().blockCount()
            if current_line_count > self.max_log_lines:
                cursor = self.log_text.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                cursor.movePosition(
                    QTextCursor.MoveOperation.Down,
                    QTextCursor.MoveMode.MoveAnchor,
                    current_line_count - self.max_log_lines
                )
                cursor.movePosition(QTextCursor.MoveOperation.Start, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
            
            # 添加新日志
            for log in new_logs:
                self._append_log_to_text(log)
            
            # 更新统计信息
            self.update_stats()
            
            # 自动滚动到底部
            if self.auto_scroll_check.isChecked():
                self.log_text.verticalScrollBar().setValue(
                    self.log_text.verticalScrollBar().maximum()
                )
    
    def should_display(self, log: dict) -> bool:
        """判断是否应该显示该日志"""
        # 级别过滤
        if self.current_filter_level != "ALL":
            level_map = {
                "DEBUG": ["DEBUG"],
                "INFO": ["INFO"],
                "WARNING": ["WARNING"],
                "ERROR": ["ERROR"],
                "CRITICAL": ["CRITICAL"]
            }
            if log['level'] not in level_map.get(self.current_filter_level, []):
                return False
        
        # 关键字过滤
        if self.filter_keyword:
            keyword = self.filter_keyword.lower()
            if keyword not in log['message'].lower() and keyword not in log['formatted'].lower():
                return False
        
        return True
    
    def _append_log_to_text(self, log: dict):
        """将日志添加到文本显示区域"""
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # 根据级别设置颜色
        color = self.get_level_color(log['level'])
        
        # 设置文本格式
        char_format = QTextCharFormat()
        char_format.setForeground(QColor(color))
        cursor.setCharFormat(char_format)
        
        # 插入文本
        cursor.insertText(log['formatted'] + "\n")
    
    def get_level_color(self, level: str) -> str:
        """获取日志级别对应的颜色"""
        colors = {
            'DEBUG': '#888888',      # 灰色
            'INFO': '#4EC9B0',       # 青色
            'WARNING': '#CE9178',    # 橙色
            'ERROR': '#F48771',      # 红色
            'CRITICAL': '#D16969',   # 深红色
        }
        return colors.get(level, '#FFFFFF')
    
    def on_level_filter_changed(self, index: int):
        """日志级别过滤改变"""
        level_map = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.current_filter_level = level_map[index] if index < len(level_map) else "ALL"
        self.refresh_all_logs()
    
    def on_search_changed(self, text: str):
        """搜索关键字改变"""
        self.filter_keyword = text
        self.refresh_all_logs()
    
    def refresh_all_logs(self):
        """刷新所有日志显示"""
        self.log_text.clear()
        
        # 重新加载所有日志
        if self.log_handler:
            buffered_logs = self.log_handler.get_buffered_logs()
            with QMutexLocker(self.mutex):
                self.log_buffer = []
                for message, level, formatted_msg in buffered_logs:
                    log = {
                        'message': message,
                        'level': level,
                        'formatted': formatted_msg,
                        'timestamp': datetime.now()
                    }
                    self.log_buffer.append(log)
                    if self.should_display(log):
                        self._append_log_to_text(log)
        
        self.update_stats()
    
    def clear_logs(self):
        """清空日志显示"""
        self.log_text.clear()
        if self.log_handler:
            self.log_handler.clear_buffer()
        with QMutexLocker(self.mutex):
            self.log_buffer.clear()
        self.update_stats()
    
    def save_logs(self):
        """保存日志到文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("保存日志"),
            f"powertools_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            self.tr("文本文件 (*.txt);;所有文件 (*)")
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    # 保存当前显示的日志
                    f.write(self.log_text.toPlainText())
                
                InfoBar.success(
                    title=self.tr("成功"),
                    content=self.tr(f"日志已保存到: {file_path}"),
                    duration=2000,
                    parent=self,
                    position=InfoBarPosition.TOP
                )
            except Exception as e:
                InfoBar.error(
                    title=self.tr("错误"),
                    content=self.tr(f"保存日志失败: {str(e)}"),
                    duration=3000,
                    parent=self,
                    position=InfoBarPosition.TOP
                )
    
    def on_font_size_changed(self, size: str):
        """字体大小改变"""
        font = self.log_text.font()
        font.setPointSize(int(size))
        self.log_text.setFont(font)
    
    def update_stats(self):
        """更新统计信息"""
        total_lines = self.log_text.document().blockCount()
        
        # 统计各级别日志数量
        level_counts = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
        
        if self.log_handler:
            buffered_logs = self.log_handler.get_buffered_logs()
            for _, level, _ in buffered_logs:
                if level in level_counts:
                    level_counts[level] += 1
        
        stats_text = f"总计: {total_lines} | " + \
                     f"DEBUG: {level_counts['DEBUG']} | " + \
                     f"INFO: {level_counts['INFO']} | " + \
                     f"WARNING: {level_counts['WARNING']} | " + \
                     f"ERROR: {level_counts['ERROR']} | " + \
                     f"CRITICAL: {level_counts['CRITICAL']}"
        
        self.stats_label.setText(stats_text)
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.refresh_timer:
            self.refresh_timer.stop()
        super().closeEvent(event)

