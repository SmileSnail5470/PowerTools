import cv2
import numpy as np
from app.algorithms import ORTEnvironment
ORTEnvironment.initialize()
from app.algorithms import general_inference_session, general_provider
from app.algorithms.visible_watermark_removal.video_modules.ppt.runtime import ppt_run_options, ppt_session_options


class RAFTBiONNX:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.prepare_model()

    def _get_session_options(self):
        return ppt_session_options()

    def __del__(self):
        if hasattr(self, 'session'):
            del self.session
            self.session = None

    def prepare_model(self):
        if self.session is not None:
            return
        sess_options = self._get_session_options()
        providers, provider_options = general_provider()
        self.session = general_inference_session(self.model_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_name_1 = self.session.get_inputs()[0].name
        self.input_name_2 = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name
        self._use_iobinding = self.session.use_cuda
        self.run_options = ppt_run_options()
        self.shrink_run_options = ppt_run_options(shrink_memory=True, use_cuda=self.session.use_cuda)

    def forward(self, gt_local_frames: np.ndarray, scale_factor: float = 1.0, shrink_memory: bool = False):
        b, l_t, c, h, w = gt_local_frames.shape
        final_run_options = (self.shrink_run_options if shrink_memory else self.run_options)
        if scale_factor < 1.0:
            new_h = int(h * scale_factor) - int(h * scale_factor) % 8
            new_w = int(w * scale_factor) - int(w * scale_factor) % 8
            frames_flat = gt_local_frames.reshape(-1, c, h, w)
            frames_ds = np.zeros((frames_flat.shape[0], c, new_h, new_w), dtype=np.float32)
            for i in range(frames_flat.shape[0]):
                img = frames_flat[i].transpose(1, 2, 0)
                frames_ds[i] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
            gt_local_frames_input = frames_ds.reshape(b, l_t, c, new_h, new_w)
        else:
            gt_local_frames_input = gt_local_frames
            new_h, new_w = h, w

        _, _, _, ih, iw = gt_local_frames_input.shape
        gtlf_1_flat = np.ascontiguousarray(gt_local_frames_input[:, :-1].reshape(-1, c, ih, iw))
        gtlf_2_flat = np.ascontiguousarray(gt_local_frames_input[:, 1:].reshape(-1, c, ih, iw))

        if self._use_iobinding:
            flow_up_forward = self.session.run_with_iobinding_numpy(
                {self.input_name_1: gtlf_1_flat, self.input_name_2: gtlf_2_flat},
                run_options=self.run_options
            )[0]
            flow_up_backward = self.session.run_with_iobinding_numpy(
                {self.input_name_1: gtlf_2_flat, self.input_name_2: gtlf_1_flat},
                run_options=final_run_options
            )[0]
        else:
            flow_up_forward = self.session.run(
                [self.output_name],
                {self.input_name_1: gtlf_1_flat, self.input_name_2: gtlf_2_flat},
                run_options=self.run_options
            )[0]
            flow_up_backward = self.session.run(
                [self.output_name],
                {self.input_name_1: gtlf_2_flat, self.input_name_2: gtlf_1_flat},
                run_options=final_run_options
            )[0]

        gt_flows_forward = flow_up_forward.reshape(b, l_t - 1, 2, ih, iw)
        gt_flows_backward = flow_up_backward.reshape(b, l_t - 1, 2, ih, iw)

        if scale_factor < 1.0:
            inv_scale = h / ih
            n_flows = gt_flows_forward.shape[1]
            flows_f_up = np.zeros((b, n_flows, 2, h, w), dtype=np.float32)
            flows_b_up = np.zeros((b, n_flows, 2, h, w), dtype=np.float32)
            for bi in range(b):
                for fi in range(n_flows):
                    flow_f_hwc = gt_flows_forward[bi, fi].transpose(1, 2, 0)
                    flow_b_hwc = gt_flows_backward[bi, fi].transpose(1, 2, 0)
                    flows_f_up[bi, fi] = (cv2.resize(flow_f_hwc, (w, h), interpolation=cv2.INTER_LINEAR) * inv_scale).transpose(2, 0, 1)
                    flows_b_up[bi, fi] = (cv2.resize(flow_b_hwc, (w, h), interpolation=cv2.INTER_LINEAR) * inv_scale).transpose(2, 0, 1)
            return flows_f_up, flows_b_up

        return gt_flows_forward, gt_flows_backward