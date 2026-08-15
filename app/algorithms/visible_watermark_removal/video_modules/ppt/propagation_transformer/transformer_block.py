import numpy as np
import onnxruntime as ort
from app.algorithms import general_inference_session, ortvalue_to_numpy
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


_ORT_TO_NP_DTYPE = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def _ortvalue_to_cupy(ort_value):
    shape = tuple(ort_value.shape())
    dtype = np.dtype(_ORT_TO_NP_DTYPE.get(ort_value.data_type(), np.float32))
    size = int(np.prod(shape)) * dtype.itemsize
    ptr = ort_value.data_ptr()
    mem = cp.cuda.UnownedMemory(ptr, size, ort_value)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    return cp.ndarray(shape, dtype=dtype, memptr=memptr)


def _cupy_to_ortvalue(arr):
    arr = cp.ascontiguousarray(arr)
    try:
        return ort.OrtValue.from_dlpack(arr.toDlpack())
    except (AttributeError, TypeError):
        return ort.OrtValue.ortvalue_from_numpy(cp.asnumpy(arr), device_type="cuda", device_id=0)


class SparseAttentionCoreORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def __call__(self, x):
        if self._use_cupy:
            x_ort = _cupy_to_ortvalue(x) if isinstance(x, cp.ndarray) else x
            feed = {self.input_names[0]: x_ort}
            ort_outputs = self.session.run_with_iobinding(feed, run_options=self.run_options)
            return (
                _ortvalue_to_cupy(ort_outputs[0]),
                _ortvalue_to_cupy(ort_outputs[1]),
                _ortvalue_to_cupy(ort_outputs[2]),
            )
        if self._use_iobinding:
            feed = {self.input_names[0]: x}
            ort_outputs = self.session.run_with_iobinding_numpy(feed, run_options=self.run_options)
            return ort_outputs[0], ort_outputs[1], ort_outputs[2]
        outputs = self.session.run(None, {self.input_names[0]: x}, run_options=self.run_options)
        return outputs[0], outputs[1], outputs[2]

    def __del__(self):
        self.session = None


class AttentionComputationORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def __call__(self, q, k, v):
        if self._use_cupy:
            feed = {
                self.input_names[0]: _cupy_to_ortvalue(q),
                self.input_names[1]: _cupy_to_ortvalue(k),
                self.input_names[2]: _cupy_to_ortvalue(v),
            }
            ort_outputs = self.session.run_with_iobinding(feed, run_options=self.run_options)
            return _ortvalue_to_cupy(ort_outputs[0])
        feed_dict = {
            self.input_names[0]: q,
            self.input_names[1]: k,
            self.input_names[2]: v,
        }
        if self._use_iobinding:
            return self.session.run_with_iobinding_numpy(feed_dict, run_options=self.run_options)[0]
        return self.session.run(None, feed_dict, run_options=self.run_options)[0]

    def __del__(self):
        self.session = None


class MLPComputationORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def __call__(self, x, enc_feat):
        if self._use_cupy:
            x_ort = _cupy_to_ortvalue(x) if isinstance(x, cp.ndarray) else x
            enc_ort = enc_feat if isinstance(enc_feat, ort.OrtValue) else _cupy_to_ortvalue(enc_feat)
            feed = {self.input_names[0]: x_ort, self.input_names[1]: enc_ort}
            ort_outputs = self.session.run_with_iobinding(feed, run_options=self.run_options)
            return _ortvalue_to_cupy(ort_outputs[0])
        feed_dict = {self.input_names[0]: x, self.input_names[1]: enc_feat}
        if self._use_iobinding:
            return self.session.run_with_iobinding(feed_dict, run_options=self.run_options)[0]
        return self.session.run(None, feed_dict, run_options=self.run_options)[0]

    def __del__(self):
        self.session = None


class OutputProjectionORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def __call__(self, att_out, x):
        if self._use_cupy:
            feed = {
                self.input_names[0]: _cupy_to_ortvalue(att_out),
                self.input_names[1]: _cupy_to_ortvalue(x),
            }
            ort_outputs = self.session.run_with_iobinding(feed, run_options=self.run_options)
            return _ortvalue_to_cupy(ort_outputs[0])
        feed_dict = {self.input_names[0]: att_out, self.input_names[1]: x}
        if self._use_iobinding:
            return self.session.run_with_iobinding(feed_dict, run_options=self.run_options)[0]
        return self.session.run(None, feed_dict, run_options=self.run_options)[0]

    def __del__(self):
        self.session = None


class NormComputationORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def __call__(self, x):
        if self._use_cupy:
            x_ort = _cupy_to_ortvalue(x) if isinstance(x, cp.ndarray) else x
            feed = {self.input_names[0]: x_ort}
            ort_outputs = self.session.run_with_iobinding(feed, run_options=self.run_options)
            return _ortvalue_to_cupy(ort_outputs[0])
        feed_dict = {self.input_names[0]: x}
        if self._use_iobinding:
            return self.session.run_with_iobinding_numpy(feed_dict, run_options=self.run_options)[0]
        return self.session.run(None, feed_dict, run_options=self.run_options)[0]

    def __del__(self):
        self.session = None


class TemporalSparseTransformerBlockORT:
    def __init__(self, depths, window_size, onnx_paths, providers, provider_options=None, sess_options=None, run_options=None):
        self.depths = depths
        self.window_size = window_size

        self.core_layers = [SparseAttentionCoreORT(onnx_paths["core"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self.attn_layers = [AttentionComputationORT(onnx_paths["attn"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self.proj_layers = [OutputProjectionORT(onnx_paths["proj"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self.norm1_layers = [NormComputationORT(onnx_paths["norm1"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self.norm2_layers = [NormComputationORT(onnx_paths["norm2"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self.mlp_layers = [MLPComputationORT(onnx_paths["mlp"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self._use_iobinding = self.proj_layers[0]._use_iobinding
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def __del__(self):
        self.core_layers = None
        self.attn_layers = None
        self.proj_layers = None
        self.norm1_layers = None
        self.norm2_layers = None
        self.mlp_layers = None

    def _numpy_max_pool2d_and_sum(self, mask, new_h, new_w, n_wh, n_ww):
        B, T, H, W, _ = mask.shape
        pad_h = new_h - H
        pad_w = new_w - W
        if pad_h > 0 or pad_w > 0:
            mask_pad = np.pad(mask, ((0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        else:
            mask_pad = mask
        mask_pad = np.squeeze(mask_pad, axis=-1)
        w_h, w_w = self.window_size
        blocked = mask_pad.reshape(B, T, n_wh, w_h, n_ww, w_w)
        pooled = blocked.max(axis=(3, 5))
        pooled_flat = pooled.reshape(B, T, n_wh * n_ww)
        return pooled_flat.sum(axis=1)

    def __call__(self, x, enc_feat, l_mask, t_dilation=2):
        if self._use_cupy:
            return self._call_cupy(x, enc_feat, l_mask, t_dilation)
        return self._call_cpu(x, enc_feat, l_mask, t_dilation)

    def _call_cpu(self, x, enc_feat, l_mask, t_dilation=2):
        assert self.depths % t_dilation == 0, 'wrong t_dilation input.'
        B, T, H, W, C = x.shape
        w_h, w_w = self.window_size
        w_h_w_w = w_h * w_w
        n_head = self.core_layers[0].session.get_outputs()[0].shape[2]
        c_head = C // n_head

        n_wh = int(np.ceil(H / w_h))
        n_ww = int(np.ceil(W / w_w))
        new_h = n_wh * w_h
        new_w = n_ww * w_w

        t_indices = []
        for _ in range(self.depths // t_dilation):
            for i in range(t_dilation):
                t_indices.append(np.arange(i, T, t_dilation))
        mask_sum = self._numpy_max_pool2d_and_sum(l_mask, new_h, new_w, n_wh, n_ww)

        if self._use_iobinding:
            enc_feat_gpu = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(enc_feat), device_type="cuda", device_id=0)
        else:
            enc_feat_gpu = enc_feat

        for i in range(self.depths):
            shortcut = x
            norm_x = self.norm1_layers[i](x)
            win_q, win_k, win_v = self.core_layers[i](norm_x)
            del norm_x

            out_layer = np.zeros_like(win_q)
            curr_t_ind = t_indices[i]
            t_sub_len = len(curr_t_ind)

            for b_idx in range(B):
                layer_mask_b = mask_sum[b_idx]
                mask_ind_i = np.nonzero(layer_mask_b != 0)[0]
                mask_n = len(mask_ind_i)
                unmask_ind_i = np.nonzero(layer_mask_b == 0)[0]
                unmask_n = len(unmask_ind_i)

                if mask_n > 0:
                    win_q_t = win_q[b_idx, mask_ind_i]
                    win_k_t = win_k[b_idx, mask_ind_i][:, :, curr_t_ind, :, :]
                    win_v_t = win_v[b_idx, mask_ind_i][:, :, curr_t_ind, :, :]
                    q_flat = win_q_t.reshape(mask_n, n_head, T * w_h_w_w, c_head)
                    k_flat = win_k_t.reshape(mask_n, n_head, t_sub_len * win_k_t.shape[3], c_head)
                    v_flat = win_v_t.reshape(mask_n, n_head, t_sub_len * win_v_t.shape[3], c_head)
                    del win_q_t, win_k_t, win_v_t
                    y_t = self.attn_layers[i](q_flat, k_flat, v_flat)
                    del q_flat, k_flat, v_flat
                    out_layer[b_idx, mask_ind_i] = y_t.reshape(mask_n, n_head, T, w_h_w_w, c_head)
                    del y_t

                if unmask_n > 0:
                    win_q_s = win_q[b_idx, unmask_ind_i]
                    win_k_s = win_k[b_idx, unmask_ind_i, :, :, :w_h_w_w, :]
                    win_v_s = win_v[b_idx, unmask_ind_i, :, :, :w_h_w_w, :]
                    q_flat_s = win_q_s.reshape(unmask_n, n_head, T * w_h_w_w, c_head)
                    k_flat_s = win_k_s.reshape(unmask_n, n_head, T * w_h_w_w, c_head)
                    v_flat_s = win_v_s.reshape(unmask_n, n_head, T * w_h_w_w, c_head)
                    del win_q_s, win_k_s, win_v_s
                    y_s = self.attn_layers[i](q_flat_s, k_flat_s, v_flat_s)
                    del q_flat_s, k_flat_s, v_flat_s
                    out_layer[b_idx, unmask_ind_i] = y_s.reshape(unmask_n, n_head, T, w_h_w_w, c_head)
                    del y_s

            del win_q, win_k, win_v
            proj_out = self.proj_layers[i](out_layer, x)
            del out_layer
            x = shortcut + ortvalue_to_numpy(proj_out)
            del shortcut, proj_out

            shortcut_mlp = x
            norm_y = self.norm2_layers[i](x)
            mlp_out = self.mlp_layers[i](
                np.ascontiguousarray(norm_y.reshape(B, T * H * W, C)),
                enc_feat_gpu
            )
            del norm_y
            x = shortcut_mlp + ortvalue_to_numpy(mlp_out).reshape(B, T, H, W, C)
            del shortcut_mlp, mlp_out

        return x

    def _call_cupy(self, x, enc_feat, l_mask, t_dilation=2):
        assert self.depths % t_dilation == 0, 'wrong t_dilation input.'
        B, T, H, W, C = x.shape
        w_h, w_w = self.window_size
        w_h_w_w = w_h * w_w
        n_head = self.core_layers[0].session.get_outputs()[0].shape[2]
        c_head = C // n_head

        n_wh = int(np.ceil(H / w_h))
        n_ww = int(np.ceil(W / w_w))
        new_h = n_wh * w_h
        new_w = n_ww * w_w

        t_indices = []
        for _ in range(self.depths // t_dilation):
            for i in range(t_dilation):
                t_indices.append(cp.arange(i, T, t_dilation))

        l_mask_np = cp.asnumpy(l_mask) if isinstance(l_mask, cp.ndarray) else l_mask
        mask_sum = self._numpy_max_pool2d_and_sum(l_mask_np, new_h, new_w, n_wh, n_ww)

        x_gpu = cp.asarray(x)
        enc_feat_ort = (
            _cupy_to_ortvalue(enc_feat)
            if isinstance(enc_feat, cp.ndarray)
            else ort.OrtValue.ortvalue_from_numpy(
                np.ascontiguousarray(enc_feat), device_type="cuda", device_id=0
            )
        )

        for i in range(self.depths):
            shortcut = x_gpu

            # norm1: cupy → OrtValue → session → cupy
            norm_x = self.norm1_layers[i](x_gpu)

            # core: cupy input → session → cupy outputs (win_q, win_k, win_v)
            win_q, win_k, win_v = self.core_layers[i](norm_x)
            del norm_x

            out_layer = cp.zeros_like(win_q)
            curr_t_ind = t_indices[i]
            t_sub_len = len(curr_t_ind)

            for b_idx in range(B):
                layer_mask_b = mask_sum[b_idx]
                mask_ind_i = np.nonzero(layer_mask_b != 0)[0]
                mask_n = len(mask_ind_i)
                unmask_ind_i = np.nonzero(layer_mask_b == 0)[0]
                unmask_n = len(unmask_ind_i)

                # --- Masked windows: temporal sparse attention ---
                if mask_n > 0:
                    # Fancy indexing on GPU
                    mask_ind_gpu = cp.asarray(mask_ind_i)
                    win_q_t = win_q[b_idx, mask_ind_gpu]
                    win_k_t = win_k[b_idx, mask_ind_gpu][:, :, curr_t_ind, :, :]
                    win_v_t = win_v[b_idx, mask_ind_gpu][:, :, curr_t_ind, :, :]
                    q_flat = win_q_t.reshape(mask_n, n_head, T * w_h_w_w, c_head)
                    k_flat = win_k_t.reshape(mask_n, n_head, t_sub_len * win_k_t.shape[3], c_head)
                    v_flat = win_v_t.reshape(mask_n, n_head, t_sub_len * win_v_t.shape[3], c_head)
                    del win_q_t, win_k_t, win_v_t

                    y_t = self.attn_layers[i](q_flat, k_flat, v_flat)
                    del q_flat, k_flat, v_flat
                    out_layer[b_idx, mask_ind_gpu] = y_t.reshape(mask_n, n_head, T, w_h_w_w, c_head)
                    del y_t, mask_ind_gpu

                # --- Unmasked windows: local attention ---
                if unmask_n > 0:
                    unmask_ind_gpu = cp.asarray(unmask_ind_i)
                    win_q_s = win_q[b_idx, unmask_ind_gpu]
                    win_k_s = win_k[b_idx, unmask_ind_gpu, :, :, :w_h_w_w, :]
                    win_v_s = win_v[b_idx, unmask_ind_gpu, :, :, :w_h_w_w, :]
                    q_flat_s = win_q_s.reshape(unmask_n, n_head, T * w_h_w_w, c_head)
                    k_flat_s = win_k_s.reshape(unmask_n, n_head, T * w_h_w_w, c_head)
                    v_flat_s = win_v_s.reshape(unmask_n, n_head, T * w_h_w_w, c_head)
                    del win_q_s, win_k_s, win_v_s

                    y_s = self.attn_layers[i](q_flat_s, k_flat_s, v_flat_s)
                    del q_flat_s, k_flat_s, v_flat_s
                    out_layer[b_idx, unmask_ind_gpu] = y_s.reshape(unmask_n, n_head, T, w_h_w_w, c_head)
                    del y_s, unmask_ind_gpu

            del win_q, win_k, win_v

            # proj on GPU: cupy → OrtValue → session → cupy
            proj_out = self.proj_layers[i](out_layer, x_gpu)
            del out_layer
            # Residual add on GPU
            x_gpu = shortcut + proj_out
            del shortcut, proj_out

            # norm2 → mlp on GPU
            shortcut_mlp = x_gpu
            norm_y = self.norm2_layers[i](x_gpu)
            mlp_out = self.mlp_layers[i](
                cp.ascontiguousarray(norm_y.reshape(B, T * H * W, C)),
                enc_feat_ort
            )
            del norm_y
            x_gpu = shortcut_mlp + mlp_out.reshape(B, T, H, W, C)
            del shortcut_mlp, mlp_out
        return x_gpu
