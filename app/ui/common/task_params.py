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
    

def bind_widget_to_param(widget, signal_name: str, param_model: TaskParams, param_key: str, transform=None):
    """通用绑定函数：控件信号驱动参数更新
    
    """
    signal = getattr(widget, signal_name)
    if transform is None:
        transform = lambda x: x

    def on_value_changed(value):
        param_model.set_param(param_key, transform(value))

    signal.connect(on_value_changed)