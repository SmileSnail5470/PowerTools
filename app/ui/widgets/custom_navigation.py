from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
from PySide6.QtGui import QFont
from app.ui.library.qfluentwidgets import setFont


class CustomNavigation(QWidget):
    def __init__(self, navigation_item_texts=[QWidget.tr("结果对比预览"), QWidget.tr("Mask对比预览")], parent=None):
        super().__init__(parent=parent)
        self.init_ui(navigation_item_texts)

    def init_ui(self, navigation_item_texts):
        self.setObjectName("CustomNavigation")
        self.setStyleSheet("#CustomNavigation { background-color: #f8fafc; }")

        self.nav_container = QFrame(self)
        self.nav_container.setContentsMargins(0, 0, 0, 0)
        self.nav_container.setObjectName("navContainer")
        self.nav_container.setStyleSheet("""
            #navContainer {
                background-color: #ffffff;
                border-radius: 14px;
                border: 1px solid rgba(226, 232, 240, 1);
            }
        """)

        self.slider = QFrame(self.nav_container)
        self.slider.setStyleSheet("""
            background-color: #eff6ff;
            border-radius: 10px;
            border: 1px solid rgba(59, 130, 246, 25);
        """)
        
        self.layout = QHBoxLayout(self.nav_container)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(4)

        self.items = []
        for text in navigation_item_texts:
            item = NavItem(text, self)
            self.items.append(item)

        for item in self.items:
            self.layout.addWidget(item)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 5, 0, 0)
        main_layout.addWidget(self.nav_container, alignment=Qt.AlignCenter)

        self.animation = QPropertyAnimation(self.slider, b"geometry")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

        self.current_active = self.items[0]
        self.items[0].set_active(True)

    def showEvent(self, event):
        super().showEvent(event)
        self.update_slider(self.items[0], animate=False)

    def update_slider(self, target, animate=True):
        if animate:
            self.animation.stop()
            self.animation.setStartValue(self.slider.geometry())
            self.animation.setEndValue(target.geometry())
            self.animation.start()
        else:
            self.slider.setGeometry(target.geometry())

class NavItem(QLabel):
    def __init__(self, text, parent_nav):
        super().__init__(text)
        self.parent_nav = parent_nav
        self.is_active = False
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(30)
        self.setContentsMargins(24, 0, 24, 0)
        setFont(self, 11, weight=QFont.Bold)
        self.update_style()

    def update_style(self):
        color = "#3b82f6" if self.is_active else "#64748b"
        self.setStyleSheet(f"color: {color}; background: transparent; border: none;")

    def set_active(self, active):
        self.is_active = active
        self.update_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            for item in self.parent_nav.items:
                item.set_active(False)
            self.set_active(True)
            self.parent_nav.update_slider(self)