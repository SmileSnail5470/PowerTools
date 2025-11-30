"""日志查看页面"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout
from app.ui.library.qfluentwidgets import ScrollArea, setFont, BodyLabel
from app.ui.widgets.log_viewer_widget import LogViewerWidget


class Logs(QWidget):
    """日志查看页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setObjectName("LogsPage")
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        # 创建滚动区域
        scroll_area = ScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # 创建内容容器
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(30, 20, 30, 30)
        container_layout.setSpacing(20)
        
        # 标题
        title = BodyLabel(self.tr("日志查看器"))
        setFont(title, 24)
        container_layout.addWidget(title)
        
        # 日志查看器组件
        self.log_viewer = LogViewerWidget(container)
        container_layout.addWidget(self.log_viewer)
        
        container_layout.addStretch()
        
        scroll_area.setWidget(container)
        
        # 设置主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)

