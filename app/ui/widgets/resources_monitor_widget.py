import os
import psutil
import time
from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QColor, QMouseEvent, QFont
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QGraphicsDropShadowEffect
import qtawesome as qta
from app.ui.library.qfluentwidgets import setFont
from app.controllers.task_manager import global_task_manager


class ResourceMonitorWorker(QObject):
    data_updated = Signal(float, float)
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = True
        self.main_pid = os.getpid()
        # 缓存进程对象：{pid: psutil.Process}
        # 必须缓存对象，否则 cpu_percent(interval=None) 无法计算差值，永远返回 0
        self.process_cache = {} 

    def run(self):
        try:
            if self.main_pid not in self.process_cache:
                self.process_cache[self.main_pid] = psutil.Process(self.main_pid)
        except psutil.NoSuchProcess:
            self.finished.emit()
            return

        while self._is_running:
            total_cpu = 0.0
            total_mem = 0.0
            
            try:
                main_proc = self.process_cache[self.main_pid]
                
                # 获取所有子进程 (递归)
                children = main_proc.children(recursive=True)
                
                # 更新进程缓存 (维护活跃进程列表)
                current_pids = {self.main_pid} | {child.pid for child in children}
                
                # 移除已终止的进程
                dead_pids = set(self.process_cache.keys()) - current_pids
                for pid in dead_pids:
                    del self.process_cache[pid]
                
                # 添加新出现的进程
                for child in children:
                    if child.pid not in self.process_cache:
                        try:
                            self.process_cache[child.pid] = child
                            # 新进程首次调用 cpu_percent 通常返回 0.0，用于初始化基准
                            child.cpu_percent(interval=None) 
                        except psutil.NoSuchProcess:
                            pass

                # 3. 计算总资源消耗
                for pid, proc in self.process_cache.items():
                    try:
                        # interval=None 是非阻塞的关键，它计算自上次调用以来的使用率
                        # 注意：多核CPU下，该值可能超过 100% (如 4核满载为 400%)
                        cpu = proc.cpu_percent(interval=None) / psutil.cpu_count()
                        
                        # rss 是物理内存 (Resident Set Size)
                        mem = proc.memory_info().rss / (1024 ** 3) # 转换为 GB
                        
                        total_cpu += cpu
                        total_mem += mem
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # 进程可能在计算过程中刚刚结束
                        pass

                # 发送数据到 UI
                self.data_updated.emit(round(total_cpu, 1), round(total_mem, 1))

            except psutil.NoSuchProcess:
                break
            except Exception:
                pass

            for _ in range(30):
                if not self._is_running:
                    break
                time.sleep(0.1)
        
        self.finished.emit()

    def stop(self):
        self._is_running = False


class ResourceItem(QFrame):
    def __init__(self, icon_name, label_text, unit="", parent=None):
        super().__init__(parent)
        self.unit = unit
        self.label_text = label_text
        
        self.setMinimumWidth(70)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("ResourceItem")
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(8, 10, 8, 10)
        self.layout.setSpacing(8)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(16, 16)
        self.icon_label.setAlignment(Qt.AlignCenter)
        icon = qta.icon(icon_name, color='#323130') 
        self.icon_label.setPixmap(icon.pixmap(16, 16))
        
        self.info_container = QWidget()
        self.info_layout = QVBoxLayout(self.info_container)
        self.info_layout.setContentsMargins(0, 0, 0, 0)
        self.info_layout.setSpacing(0)
        
        self.value_label = QLabel("0" + unit)
        setFont(self.value_label, 11, QFont.Bold)
        self.value_label.setObjectName("ResourceValue")
        self.value_label.setAlignment(Qt.AlignLeft)
        
        self.title_label = QLabel(label_text)
        setFont(self.title_label, 9)
        self.title_label.setObjectName("ResourceLabel")
        self.title_label.setAlignment(Qt.AlignLeft)
        
        self.info_layout.addWidget(self.value_label)
        self.info_layout.addWidget(self.title_label)

        self.status_dot = QFrame()
        self.status_dot.setFixedSize(6, 6)
        self.status_dot.setStyleSheet("border-radius: 3px; background-color: #107c10;")
        
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.info_container)
        self.layout.addWidget(self.status_dot)
        self.layout.addStretch() # 靠左对齐，右侧填充

    def update_data(self, value, raw_value):
        self.value_label.setText(f"{value}{self.unit}")
        
        if raw_value is None or raw_value < 50:
            color = "#107c10"
            shadow = "rgba(16, 124, 16, 0.2)"
        elif raw_value < 80:
            color = "#ff8c00"
            shadow = "rgba(255, 140, 0, 0.2)"
        else:
            color = "#d13438"
            shadow = "rgba(209, 52, 56, 0.2)"
            
        self.status_dot.setStyleSheet(f"""
            background-color: {color};
            border: 2px solid {shadow};
            border-radius: 3px;
        """)

class VerticalDivider(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFixedSize(2, 24)
        self.setStyleSheet("background-color: transparent; border: none;")


class ResourcesMonitorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.container = QFrame()
        self.container.setObjectName("ToolbarContainer")
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.container)
        
        self.toolbar_layout = QHBoxLayout(self.container)
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar_layout.setSpacing(4)

        self.cpu_item = ResourceItem('fa5s.microchip', 'CPU', '%')
        self.mem_item = ResourceItem('fa5s.memory', '内存', 'GB')
        self.task_item = ResourceItem('fa5s.tasks', '活跃任务', '')

        self.toolbar_layout.addWidget(self.cpu_item)
        self.toolbar_layout.addWidget(VerticalDivider())
        self.toolbar_layout.addWidget(self.mem_item)
        self.toolbar_layout.addWidget(VerticalDivider())
        self.toolbar_layout.addWidget(self.task_item)

        self.setup_styles()

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.container.setGraphicsEffect(shadow)

        self._start_monitor()

        self.total_mem = psutil.virtual_memory().total / (1024 ** 3)

    def setup_styles(self):
        style_sheet = """
        #ToolbarContainer {
            background-color: transparent;
            border: 1px solid rgba(255, 255, 255, 80);
            border-radius: 8px;
        }

        #ResourceItem {
            border-radius: 4px;
            background-color: transparent;
        }
        #ResourceItem:hover {
            background-color: rgba(0, 120, 212, 20); /* 0.08 alpha */
        }
        #ResourceItem:pressed {
            background-color: rgba(0, 120, 212, 30); /* 0.12 alpha */
        }

        #ResourceValue {
            color: #323130;
            line-height: 14px;
        }
        
        #ResourceLabel {
            color: #605e5c;
            line-height: 10px;
        }
        """
        self.setStyleSheet(style_sheet)

    def update_resources(self, cpu_percent: float, mem_mb: float):
        self.cpu_item.update_data(cpu_percent, cpu_percent)
        
        self.mem_item.update_data(mem_mb, mem_mb / self.total_mem * 100)
        
        task_val = global_task_manager.get_active_task_count()
        global_task_manager.cleanup_completed_tasks()
        self.task_item.update_data(task_val, None)

    def _start_monitor(self):
        self.monitor_thread = QThread(self)
        self.worker = ResourceMonitorWorker()
        self.worker.moveToThread(self.monitor_thread)
        
        # 连接信号
        self.monitor_thread.started.connect(self.worker.run)
        self.worker.data_updated.connect(self.update_resources)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.monitor_thread.quit)
        self.monitor_thread.finished.connect(self.monitor_thread.deleteLater)
        
        self.monitor_thread.start()

    def clear(self):
        self.worker.stop()
        self.monitor_thread.quit()
        self.monitor_thread.wait()
        self.worker = None
        self.monitor_thread = None
        