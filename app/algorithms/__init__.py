import threading
import onnxruntime as ort


class ORTEnvironment:
    _initialized = False
    _lock = threading.Lock()

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return
        with cls._lock:
            if cls._initialized:
                return
            info = ort.OrtMemoryInfo("Cpu", ort.OrtAllocatorType.ORT_DEVICE_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
            ort.create_and_register_allocator(info, None)
            cls._initialized = True

