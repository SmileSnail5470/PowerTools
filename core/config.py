import sys
from app.ui.library.qfluentwidgets import QConfig


class UpdateFontFamilies:
    """"根据系统选择合适字体.
    
    Windows: Segoe UI, Microsoft YaHei
    MacOS: SF Pro, PingFang SC
    Linux: Ubuntu, Noto Sans CJK SC

    字体结果会写到调用进程的环境变量 POWWETOOLS_FONT_FAMILIES 中.
    """
    def __init__(self, cfg: QConfig):
        self.platform = sys.platform
        self.cfg = cfg

    def run(self):
        if self.platform.startswith('win'):
            families = ['Segoe UI', 'Microsoft YaHei']
        elif self.platform.startswith('darwin'):
            families = ['PingFang SC', 'Microsoft YaHei']
        else:
            families = ['Noto Sans CJK SC', 'Microsoft YaHei']
        self.cfg.fontFamilies.value = families