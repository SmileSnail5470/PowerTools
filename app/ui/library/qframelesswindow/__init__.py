import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QMainWindow

from .titlebar import TitleBar, TitleBarButton, SvgTitleBarButton, StandardTitleBar, TitleBarBase

if sys.platform == "win32":
    from .windows import AcrylicWindow
    from .windows import WindowsFramelessWindow as FramelessWindow
    from .windows import WindowsFramelessMainWindow as FramelessMainWindow
    from .windows import WindowsFramelessDialog as FramelessDialog
    from .windows import WindowsWindowEffect as WindowEffect
elif sys.platform == "darwin":
    from .mac import AcrylicWindow
    from .mac import MacFramelessWindow as FramelessWindow
    from .mac import MacFramelessMainWindow as FramelessMainWindow
    from .mac import MacFramelessDialog as FramelessDialog
    from .mac import MacWindowEffect as WindowEffect
else:
    from .linux import LinuxFramelessWindow as FramelessWindow
    from .linux import LinuxFramelessMainWindow as FramelessMainWindow
    from .linux import LinuxFramelessDialog as FramelessDialog
    from .linux import LinuxWindowEffect as WindowEffect

    AcrylicWindow = FramelessWindow

