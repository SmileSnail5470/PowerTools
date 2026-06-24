import json
import os
import pathlib
import threading
import onnxruntime as ort
import app.library._model_loader as model_loader
from app.ui.common.utils import global_backend_info_cache


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
            if "GPU" in global_backend_info_cache.get()[0]:
                cuda_info = ort.OrtMemoryInfo("Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
                arena_cfg = ort.OrtArenaCfg(0, 1, -1, -1)
                ort.create_and_register_allocator_v2("CUDAExecutionProvider", cuda_info, {}, arena_cfg)
            info = ort.OrtMemoryInfo("Cpu", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
            ort.create_and_register_allocator(info, None)
            cls._initialized = True


def general_inference_session(model_path: str, sess_options, providers, provider_options):
    model_name = os.environ["_feature_name_"]
    lic_path = os.path.join(os.path.join(pathlib.Path.home(), ".PowerTools", "license"), "license.lic")
    with open(lic_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    license_json = json.dumps(raw_data)
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(model_path))
    sess = model_loader.load_model_auto(
        model_path,
        model_name, 
        license_json, 
        session_options=sess_options, 
        providers=providers, 
        provider_options=provider_options
    )
    os.chdir(old_cwd)
    return sess

