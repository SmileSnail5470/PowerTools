import os
import platform
import sys
import numpy as np
import onnxruntime as ort
from app.algorithms import ORTEnvironment
ORTEnvironment.initialize()
from app.algorithms import general_inference_session, general_session, general_provider


class RAFTBiONNX:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.prepare_model()

    def _get_session_options(self):
        opts = general_session()
        return opts
    
    def __del__(self):
        if hasattr(self, 'session'):
            del self.session
            self.session = None
    
    def prepare_model(self):
        if self.session is not None:
            return
        sess_options = self._get_session_options()
        providers, provider_options = general_provider()
        self.run_options = ort.RunOptions()
        self.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
        self.session = general_inference_session(self.model_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_name_1 = self.session.get_inputs()[0].name
        self.input_name_2 = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, gt_local_frames: np.ndarray):
        b, l_t, c, h, w = gt_local_frames.shape

        gtlf_1 = gt_local_frames[:, :-1, :, :, :]
        gtlf_2 = gt_local_frames[:, 1:, :, :, :]

        gtlf_1_flat = gtlf_1.reshape(-1, c, h, w)
        gtlf_2_flat = gtlf_2.reshape(-1, c, h, w)
        gtlf_1_flat = np.ascontiguousarray(gtlf_1_flat)
        gtlf_2_flat = np.ascontiguousarray(gtlf_2_flat)

        outputs_forward = self.session.run(
            [self.output_name],
            {self.input_name_1: gtlf_1_flat, self.input_name_2: gtlf_2_flat},
            run_options=self.run_options
        )
        flow_up_forward = outputs_forward[0]

        outputs_backward = self.session.run(
            [self.output_name],
            {self.input_name_1: gtlf_2_flat, self.input_name_2: gtlf_1_flat},
            run_options=self.run_options
        )
        flow_up_backward = outputs_backward[0]

        gt_flows_forward = flow_up_forward.reshape(b, l_t - 1, 2, h, w)
        gt_flows_backward = flow_up_backward.reshape(b, l_t - 1, 2, h, w)
        return gt_flows_forward, gt_flows_backward