import cv2
import numpy as np
import onnxruntime as ort
from app.algorithms import ORTEnvironment
ORTEnvironment.initialize()
from app.algorithms import general_inference_session, general_session, general_provider, is_gpu_device


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
        self.session = general_inference_session(self.model_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_name_1 = self.session.get_inputs()[0].name
        self.input_name_2 = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name
        self._use_iobinding = self.session.use_cuda
        self.run_options = ort.RunOptions()
        if is_gpu_device():
            self.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
        else:
            self.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")

    def forward(self, gt_local_frames: np.ndarray, scale_factor: float = 1.0):
        b, l_t, c, h, w = gt_local_frames.shape
        if scale_factor < 1.0:
            new_h = int(h * scale_factor) - int(h * scale_factor) % 8
            new_w = int(w * scale_factor) - int(w * scale_factor) % 8
            frames_flat = gt_local_frames.reshape(-1, c, h, w)
            frames_ds = np.zeros((frames_flat.shape[0], c, new_h, new_w), dtype=np.float32)
            for i in range(frames_flat.shape[0]):
                img = frames_flat[i].transpose(1, 2, 0)
                img_ds = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                frames_ds[i] = img_ds.transpose(2, 0, 1)
            gt_local_frames_input = frames_ds.reshape(b, l_t, c, new_h, new_w)
        else:
            gt_local_frames_input = gt_local_frames
            new_h, new_w = h, w

        _, _, _, ih, iw = gt_local_frames_input.shape

        gtlf_1 = gt_local_frames_input[:, :-1, :, :, :]
        gtlf_2 = gt_local_frames_input[:, 1:, :, :, :]

        gtlf_1_flat = np.ascontiguousarray(gtlf_1.reshape(-1, c, ih, iw))
        gtlf_2_flat = np.ascontiguousarray(gtlf_2.reshape(-1, c, ih, iw))

        if self._use_iobinding:
            outputs_forward = self.session.run_with_iobinding_numpy(
                {self.input_name_1: gtlf_1_flat, self.input_name_2: gtlf_2_flat},
                run_options=self.run_options
            )
            flow_up_forward = outputs_forward[0]

            outputs_backward = self.session.run_with_iobinding_numpy(
                {self.input_name_1: gtlf_2_flat, self.input_name_2: gtlf_1_flat},
                run_options=self.run_options
            )
            flow_up_backward = outputs_backward[0]
        else:
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

        gt_flows_forward = flow_up_forward.reshape(b, l_t - 1, 2, ih, iw)
        gt_flows_backward = flow_up_backward.reshape(b, l_t - 1, 2, ih, iw)

        if scale_factor < 1.0:
            inv_scale = h / ih
            n_flows = gt_flows_forward.shape[1]
            flows_f_up = np.zeros((b, n_flows, 2, h, w), dtype=np.float32)
            flows_b_up = np.zeros((b, n_flows, 2, h, w), dtype=np.float32)
            for bi in range(b):
                for fi in range(n_flows):
                    for ch in range(2):
                        flows_f_up[bi, fi, ch] = cv2.resize(
                            gt_flows_forward[bi, fi, ch], (w, h), interpolation=cv2.INTER_LINEAR
                        ) * inv_scale
                        flows_b_up[bi, fi, ch] = cv2.resize(
                            gt_flows_backward[bi, fi, ch], (w, h), interpolation=cv2.INTER_LINEAR
                        ) * inv_scale
            return flows_f_up, flows_b_up

        return gt_flows_forward, gt_flows_backward