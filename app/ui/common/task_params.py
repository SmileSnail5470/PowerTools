from copy import deepcopy
from PySide6.QtCore import QObject, Signal


class TaskParams(QObject):
    param_changed = Signal(str, object)

    def __init__(self, **kwargs):
        super().__init__()
        self._params = kwargs or {}

    def set_param(self, key: str, value):
        old = self._params.get(key)
        if old != value:
            self._params[key] = value
            self.param_changed.emit(key, value)

    def get_param(self, key: str, default=None):
        return self._params.get(key, default)

    def to_dict(self):
        return dict(self._params)


class LayeredTaskParams(QObject):
    param_changed = Signal(str, object)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def get_param(self, path: str):
        value = self.params
        for name in path.split("."):
            value = getattr(value, name)
        return value

    def set_param(self, path: str, value) -> bool:
        parts = path.split(".")
        obj = self.params

        for name in parts[:-1]:
            obj = getattr(obj, name)

        name = parts[-1]
        old_value = getattr(obj, name)

        if old_value == value:
            return False

        setattr(obj, name, value)
        self.param_changed.emit(path, value)
        return True

    def validate(self) -> str:
        return self.params.validate()

    def snapshot(self):
        return deepcopy(self.params)
    

def bind_widget_to_param(widget, signal_name: str, param_model: TaskParams | LayeredTaskParams, param_key: str, transform=None):
    """通用绑定函数：控件信号驱动参数更新
    
    """
    signal = getattr(widget, signal_name)

    def on_value_changed(*args):
        value = transform(*args) if transform else args[0] if args else None
        param_model.set_param(param_key, value)

    signal.connect(on_value_changed)