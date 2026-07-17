import onnxruntime as ort
from app.algorithms import general_session


def ppt_session_options():
    options = general_session()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    return options


def ppt_run_options(shrink_memory=False, use_cuda=False):
    options = ort.RunOptions()
    if shrink_memory:
        device = "gpu:0" if use_cuda else "cpu"
        options.add_run_config_entry(
            "memory.enable_memory_arena_shrinkage", device
        )
    return options
