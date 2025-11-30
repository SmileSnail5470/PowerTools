import logging
from typing import Any, Dict, Callable
from app.ui.common.utils import get_file_type
from app.utils.logger.decorators import log_performance

TYPE_CASTERS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": lambda x: str(x).lower() in ("1", "true", "yes"),
    "list": lambda x: list(x),
    "dict": lambda x: dict(x),
}

class BaseWorker():
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def to_worker(self):
        """转换成 TaskManager 可用的 Worker
        """
        def task_func(*args, progress_cb=None, cancel_requested=None, **kwargs):
            if cancel_requested and cancel_requested():
                raise InterruptedError("Task was cancelled before start")

            result = self.run_algorithm(progress_cb, cancel_requested, *args, **kwargs)

            if cancel_requested and cancel_requested():
                raise InterruptedError("Task was cancelled after execution")
            return result

        return task_func, self.args, self.kwargs
    
    def file_type(self, input_file):
        return get_file_type(input_file=input_file)

    def _load_python_callable(self, instance: Any, config: dict) -> Callable:
        method_name = config.get("method")
        callable_obj = getattr(instance, method_name)
        return callable_obj

    def _cast_args(self, arg_schema: Dict[str, str], args: Dict[str, Any]) -> Dict[str, Any]:
        casted = {}
        for key, value in args.items():
            if key not in arg_schema:
                raise ValueError(f"Unsupported argument: {key}")
            type_name = arg_schema[key]
            caster = TYPE_CASTERS.get(type_name, lambda x: x)
            casted[key] = caster(value)
        return casted

    @log_performance(logger=logging.getLogger('performance'), threshold=0.1, log_args=False, log_result=False)
    def call_algorithm(self, instance: Any, method_metadata: dict, input_args: dict):
        callable_method = self._load_python_callable(instance, method_metadata)
        schema = method_metadata.get("method_args", {})
        casted_args = self._cast_args(schema, input_args)

        return callable_method(**casted_args)