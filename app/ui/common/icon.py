# coding: utf-8
from enum import Enum

from app.ui.library.qfluentwidgets import FluentIconBase, getIconColor, Theme, theme


class Icon(FluentIconBase, Enum):

    OCR = "ocr"
    IMAGE_EDIT = "image_edit"
    WATERMARK_ADD = "watermark_add"
    WATERMARK_REMOVE = "watermark_remove"
    LONG_SCREENSHOT = "long_screenshot"
    SCREENSHOT = "screenshot"

    Price = "Price"

    def path(self, theme=Theme.AUTO):
        return f":/powertools/images/icons/{self.value}_{getIconColor(theme)}.svg"
    
def icon_path(icon_name):
    return f":/powertools/images/icons/{icon_name}_{getIconColor(theme())}.svg"
