import gc
import os
import time
import math
import cv2
import logging
import numpy as np
import scipy.ndimage
from PIL import Image
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from app.algorithms import evict_session_cache
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.run_propagation_transformer import PropagationTransformerORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.raft.run_raft import RAFTBiONNX
from app.algorithms.visible_watermark_removal.video_modules.ppt.recurrent_flow_completion.run_flow_completion import RecurrentFlowCompleteORT
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


ppt_logger = logging.getLogger("subprocess")


class PPTInferenceORT:
    def __init__(
        self,
        onnx_paths,
        resize_ratio=1.0,
        height=-1,
        width=-1,
        mask_dilation=4,
        ref_stride=10,
        neighbor_length=10,
        subvideo_length=80,
        low_memory=True,
        chunk_size=600,
        chunk_overlap=40,
    ):
        self.onnx_paths = onnx_paths
        self.resize_ratio = resize_ratio
        self.height = height
        self.width = width
        self.mask_dilation = mask_dilation
        self.ref_stride = ref_stride
        self.neighbor_length = neighbor_length
        self.subvideo_length = subvideo_length
        self.low_memory = low_memory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.raft_model = None
        self.fix_flow_complete = None
        self.ppt_pipeline = None
        self._use_cupy = _HAS_CUPY

    @staticmethod
    def _flatten_model_paths(value):
        if isinstance(value, (str, os.PathLike)):
            return [os.fspath(value)]
        if isinstance(value, dict):
            paths = []
            for child in value.values():
                paths.extend(PPTInferenceORT._flatten_model_paths(child))
            return paths
        if isinstance(value, (list, tuple, set)):
            paths = []
            for child in value:
                paths.extend(PPTInferenceORT._flatten_model_paths(child))
            return paths
        return []

    def _flow_complete_model_paths(self):
        base_dir = self.onnx_paths["recurrent_flow_complete"]
        return [
            os.path.join(base_dir, name)
            for name in (
                "encoder.encmodel",
                "backward_step.encmodel",
                "forward_step.encmodel",
                "backward_backbone.encmodel",
                "forward_backbone.encmodel",
                "fusion.encmodel",
                "decoder.encmodel",
            )
        ]

    def _prepare_raft(self):
        if self.raft_model is None:
            self.raft_model = RAFTBiONNX(self.onnx_paths["raft"])

    def _prepare_flow_complete(self):
        if self.fix_flow_complete is None:
            self.fix_flow_complete = RecurrentFlowCompleteORT(onnx_dir=self.onnx_paths["recurrent_flow_complete"])

    def _prepare_ppt(self):
        if self.ppt_pipeline is None:
            self.ppt_pipeline = PropagationTransformerORT(self.onnx_paths["ppt"])
            self._use_cupy = getattr(self.ppt_pipeline, "_use_cupy", False) and _HAS_CUPY

    @staticmethod
    def _trim_device_memory():
        if not _HAS_CUPY:
            return
        try:
            cp.cuda.get_current_stream().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _release_stage(self, attr_name, model_paths):
        model = getattr(self, attr_name, None)
        had_model = model is not None
        if had_model:
            setattr(self, attr_name, None)
            del model
        evicted = evict_session_cache(model_paths) if self.low_memory else 0
        if had_model or evicted:
            gc.collect()
            if self.low_memory:
                self._trim_device_memory()

    def release(self):
        self._release_stage("raft_model", [self.onnx_paths["raft"]])
        self._release_stage("fix_flow_complete", self._flow_complete_model_paths())
        self._release_stage("ppt_pipeline", self._flatten_model_paths(self.onnx_paths["ppt"]))
        self._use_cupy = False

    def _imwrite(self, bgr_img, file_path, params=None):
        os.makedirs(os.path.abspath(os.path.dirname(file_path)), exist_ok=True)
        return cv2.imwrite(file_path, bgr_img, params)

    def _write_output_frame(self, idx, rgb_frame, out_size, output_dir):
        frame = cv2.resize(rgb_frame, out_size, interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._imwrite(frame, os.path.join(output_dir, self.frames_name[idx]))

    def _queue_output_frame(self, writer, pending, idx, frame, out_size, output_dir):
        pending.append(writer.submit(self._write_output_frame, idx, frame, out_size, output_dir))
        if len(pending) >= 2:
            pending.popleft().result()

    def _binary_mask(self, mask, th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        return mask

    def _compute_sizes(self, input_frames_dir):
        first_name = self.frames_name[0]
        first_frame = cv2.imread(os.path.join(input_frames_dir, first_name))
        h, w = first_frame.shape[:2]
        out_size = (w, h)
        size = out_size
        if self.width != -1 and self.height != -1:
            size = (self.width, self.height)
        if self.resize_ratio != 1.0:
            size = (int(self.resize_ratio * size[0]), int(self.resize_ratio * size[1]))
        process_size = (size[0] - size[0] % 8, size[1] - size[1] % 8)
        return process_size, out_size

    def _load_frames_range(self, input_frames_dir, start, end, process_size):
        pw, ph = process_size
        count = end - start
        frames_np = np.empty((1, count, 3, ph, pw), dtype=np.float32)
        for i in range(count):
            path = os.path.join(input_frames_dir, self.frames_name[start + i])
            frame = cv2.imread(path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame.shape[1], frame.shape[0]) != (pw, ph):
                frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_LINEAR)
            frames_np[0, i] = frame.transpose(2, 0, 1).astype(np.float32) * (2.0 / 255.0) - 1.0
        return frames_np

    def _load_ori_frames_range(self, input_frames_dir, start, end, process_size):
        pw, ph = process_size
        count = end - start
        ori = np.empty((count, ph, pw, 3), dtype=np.uint8)
        for i in range(count):
            path = os.path.join(input_frames_dir, self.frames_name[start + i])
            frame = cv2.imread(path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame.shape[1], frame.shape[0]) != (pw, ph):
                frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_LINEAR)
            ori[i] = frame
        return ori

    def _load_masks_range(self, masks_dir, start, end, process_size, dilates=4):
        pw, ph = process_size
        count = end - start
        if os.path.isfile(masks_dir):
            mask_img = Image.open(masks_dir).resize((pw, ph), Image.NEAREST)
            mask_arr = np.array(mask_img.convert("L"))
            f_mask = (scipy.ndimage.binary_dilation(mask_arr, iterations=dilates).astype(np.uint8) if dilates > 0 else self._binary_mask(mask_arr.copy()).astype(np.uint8))
            m_mask = (scipy.ndimage.binary_dilation(mask_arr, iterations=dilates).astype(np.uint8) if dilates > 0 else self._binary_mask(mask_arr.copy()).astype(np.uint8))
            flow_masks_np = np.broadcast_to((f_mask > 0).astype(np.uint8)[None, None, None, :, :], (1, count, 1, ph, pw),).copy()
            masks_dilated_np = np.broadcast_to((m_mask > 0).astype(np.uint8)[None, None, None, :, :], (1, count, 1, ph, pw),).copy()
        else:
            mask_names = sorted(os.listdir(masks_dir))
            if len(mask_names) == 1:
                mask_img = Image.open(os.path.join(masks_dir, mask_names[0])).resize((pw, ph), Image.NEAREST)
                mask_arr = np.array(mask_img.convert("L"))
                f_mask = (scipy.ndimage.binary_dilation(mask_arr, iterations=dilates).astype(np.uint8) if dilates > 0 else self._binary_mask(mask_arr.copy()).astype(np.uint8))
                m_mask = (scipy.ndimage.binary_dilation(mask_arr, iterations=dilates).astype(np.uint8) if dilates > 0 else self._binary_mask(mask_arr.copy()).astype(np.uint8))
                flow_masks_np = np.broadcast_to((f_mask > 0).astype(np.uint8)[None, None, None, :, :], (1, count, 1, ph, pw),).copy()
                masks_dilated_np = np.broadcast_to((m_mask > 0).astype(np.uint8)[None, None, None, :, :], (1, count, 1, ph, pw),).copy()
            else:
                flow_masks_np = np.empty((1, count, 1, ph, pw), dtype=np.uint8)
                masks_dilated_np = np.empty((1, count, 1, ph, pw), dtype=np.uint8)
                for i in range(count):
                    idx = start + i
                    mname = mask_names[idx] if idx < len(mask_names) else mask_names[-1]
                    mask_img = Image.open(os.path.join(masks_dir, mname)).resize((pw, ph), Image.NEAREST)
                    mask_arr = np.array(mask_img.convert("L"))
                    f_mask = (scipy.ndimage.binary_dilation(mask_arr, iterations=dilates).astype(np.uint8) if dilates > 0 else self._binary_mask(mask_arr.copy()).astype(np.uint8))
                    m_mask = (scipy.ndimage.binary_dilation(mask_arr, iterations=dilates).astype(np.uint8) if dilates > 0 else self._binary_mask(mask_arr.copy()).astype(np.uint8))
                    flow_masks_np[0, i, 0] = (f_mask > 0).astype(np.uint8)
                    masks_dilated_np[0, i, 0] = (m_mask > 0).astype(np.uint8)
        return np.ascontiguousarray(flow_masks_np), np.ascontiguousarray(masks_dilated_np)

    def _plan_chunks(self, total_frames):
        overlap = self.chunk_overlap
        target_core = self.chunk_size - overlap

        if total_frames <= self.chunk_size:
            return [(0, total_frames, 0, total_frames)]

        n_chunks = max(1, math.ceil((total_frames - overlap) / target_core))
        core_size = math.ceil((total_frames - overlap) / n_chunks)
        chunks = []
        for i in range(n_chunks):
            core_start = i * core_size
            core_end = min(total_frames, core_start + core_size)
            if i == n_chunks - 1:
                core_end = total_frames
            half_overlap = overlap // 2
            chunk_start = max(0, core_start - half_overlap)
            chunk_end = min(total_frames, core_end + half_overlap)
            if i == 0:
                chunk_start = 0
            if i == n_chunks - 1:
                chunk_end = total_frames
            chunks.append((chunk_start, chunk_end, core_start, core_end))
        return chunks

    def inference(self, input_frames_dir, masks_dir, output_dir, debug=True):
        try:
            return self._inference_impl(input_frames_dir, masks_dir, output_dir, debug=debug)
        finally:
            self.release()

    def _inference_impl(self, input_frames_dir, masks_dir, output_dir, debug=True):
        self.frames_name = sorted(os.listdir(input_frames_dir))
        total_frames = len(self.frames_name)
        process_size, out_size = self._compute_sizes(input_frames_dir)
        os.makedirs(output_dir, exist_ok=True)

        chunks = self._plan_chunks(total_frames)
        if debug:
            ppt_logger.info(
                f"Streaming pipeline: {total_frames} total frames, "
                f"{len(chunks)} chunks, chunk_size={self.chunk_size}, "
                f"overlap={self.chunk_overlap}, process_size={process_size}, "
                f"low memory: {self.low_memory} and use cupy {self._use_cupy}"
            )
        for chunk_idx, (chunk_start, chunk_end, core_start, core_end) in enumerate(chunks):
            chunk_len = chunk_end - chunk_start
            local_core_start = core_start - chunk_start
            local_core_end = core_end - chunk_start
            if debug:
                ppt_logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)}: frames [{chunk_start}, {chunk_end}) core [{core_start}, {core_end}) chunk_len={chunk_len}")
                chunk_time = time.time()
            self._process_chunk(
                input_frames_dir=input_frames_dir,
                masks_dir=masks_dir,
                output_dir=output_dir,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                local_core_start=local_core_start,
                local_core_end=local_core_end,
                global_core_start=core_start,
                process_size=process_size,
                out_size=out_size,
                is_last_chunk=(chunk_idx == len(chunks) - 1),
                debug=debug,
            )
            if debug:
                ppt_logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)} done in {time.time() - chunk_time:.2f}s")

    def _process_chunk(
        self,
        input_frames_dir,
        masks_dir,
        output_dir,
        chunk_start,
        chunk_end,
        local_core_start,
        local_core_end,
        global_core_start,
        process_size,
        out_size,
        is_last_chunk,
        debug,
    ):
        chunk_len = chunk_end - chunk_start
        pw, ph = process_size
        frames_np = self._load_frames_range(input_frames_dir, chunk_start, chunk_end, process_size)
        flow_masks_np, masks_dilated_np = self._load_masks_range(masks_dir, chunk_start, chunk_end, process_size, dilates=self.mask_dilation)
        if debug:
            start_time = time.time()
        self._prepare_raft()
        raft_scale = 0.5 if min(ph, pw) >= 240 else 1.0
        short_clip_len = 12 if max(ph, pw) <= 1280 else 6
        if chunk_len > short_clip_len:
            flow_shape = (1, chunk_len - 1, 2, ph, pw)
            gt_flows_f = np.empty(flow_shape, dtype=np.float32)
            gt_flows_b = np.empty(flow_shape, dtype=np.float32)
            flow_offset = 0
            for frame_start in range(0, chunk_len, short_clip_len):
                frame_end = min(chunk_len, frame_start + short_clip_len)
                sub_frames = (frames_np[:, frame_start:frame_end] if frame_start == 0 else frames_np[:, frame_start - 1:frame_end])
                flows_f, flows_b = self.raft_model.forward(sub_frames, scale_factor=raft_scale, shrink_memory=(frame_end == chunk_len))
                flow_count = flows_f.shape[1]
                gt_flows_f[:, flow_offset:flow_offset + flow_count] = flows_f
                gt_flows_b[:, flow_offset:flow_offset + flow_count] = flows_b
                flow_offset += flow_count
                del flows_f, flows_b
            if flow_offset != chunk_len - 1:
                raise RuntimeError(f"RAFT produced {flow_offset} flows for {chunk_len} frames in chunk")
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = self.raft_model.forward(frames_np, scale_factor=raft_scale, shrink_memory=True)
        if debug:
            ppt_logger.info(f"RAFT cost: {time.time() - start_time:.4f}s")
            start_time = time.time()

        self._prepare_flow_complete()
        flow_length = gt_flows_bi[0].shape[1]
        if flow_length > self.subvideo_length:
            pred_flows_f = np.empty_like(gt_flows_bi[0])
            pred_flows_b = np.empty_like(gt_flows_bi[1])
            pad_len = 5
            for flow_start in range(0, flow_length, self.subvideo_length):
                slice_start = max(0, flow_start - pad_len)
                slice_end = min(flow_length, flow_start + self.subvideo_length + pad_len)
                left_pad = flow_start - slice_start
                right_pad = slice_end - min(flow_length, flow_start + self.subvideo_length)
                sub_f, sub_b = self.fix_flow_complete.forward_bidirect_flow(
                    gt_flows_bi[0][:, slice_start:slice_end],
                    gt_flows_bi[1][:, slice_start:slice_end],
                    flow_masks_np[:, slice_start:slice_end + 1],
                    shrink_memory=(flow_start + self.subvideo_length >= flow_length),
                )
                sub_comb_f, sub_comb_b = self.fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, slice_start:slice_end], gt_flows_bi[1][:, slice_start:slice_end]),
                    (sub_f, sub_b),
                    flow_masks_np[:, slice_start:slice_end + 1],
                )
                core_end_fc = slice_end - slice_start - right_pad
                core_f = sub_comb_f[:, left_pad:core_end_fc]
                core_b = sub_comb_b[:, left_pad:core_end_fc]
                dst_end = flow_start + core_f.shape[1]
                pred_flows_f[:, flow_start:dst_end] = core_f
                pred_flows_b[:, flow_start:dst_end] = core_b
                del sub_f, sub_b, sub_comb_f, sub_comb_b, core_f, core_b
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_f, pred_flows_b = self.fix_flow_complete.forward_bidirect_flow(
                gt_flows_bi[0],
                gt_flows_bi[1],
                flow_masks_np,
                shrink_memory=True,
            )
            pred_flows_bi = self.fix_flow_complete.combine_flow(
                gt_flows_bi,
                (pred_flows_f, pred_flows_b),
                flow_masks_np,
            )
            del pred_flows_f, pred_flows_b

        del gt_flows_bi, flow_masks_np
        gc.collect()
        if debug:
            ppt_logger.info(f"Flow Completion cost: {time.time() - start_time:.4f}s")
            start_time = time.time()

        for fs in range(0, chunk_len, self.subvideo_length):
            fe = min(chunk_len, fs + self.subvideo_length)
            np.multiply(frames_np[:, fs:fe], masks_dilated_np[:, fs:fe] == 0, out=frames_np[:, fs:fe])

        masked_frames_np = frames_np
        frames_np = None
        self._prepare_ppt()
        subvideo_length_img_prop = min(100, self.subvideo_length)
        if chunk_len > subvideo_length_img_prop:
            updated_frames = np.empty_like(masked_frames_np)
            updated_masks = np.empty(masks_dilated_np.shape, dtype=np.float32)
            pad_len = 10
            for frame_start in range(0, chunk_len, subvideo_length_img_prop):
                slice_start = max(0, frame_start - pad_len)
                slice_end = min(chunk_len, frame_start + subvideo_length_img_prop + pad_len)
                left_pad = frame_start - slice_start
                right_pad = slice_end - min(chunk_len, frame_start + subvideo_length_img_prop)
                sub_flows_bi = (
                    pred_flows_bi[0][:, slice_start:slice_end - 1],
                    pred_flows_bi[1][:, slice_start:slice_end - 1],
                )
                prop_imgs_sub, updated_masks_sub = self.ppt_pipeline.img_propagation(
                    masked_frames_np[:, slice_start:slice_end],
                    sub_flows_bi,
                    masks_dilated_np[:, slice_start:slice_end],
                    shrink_memory=(frame_start + subvideo_length_img_prop >= chunk_len),
                )
                np.multiply(
                    prop_imgs_sub,
                    masks_dilated_np[:, slice_start:slice_end],
                    out=prop_imgs_sub,
                )
                prop_imgs_sub += masked_frames_np[:, slice_start:slice_end]
                core_end_ip = slice_end - slice_start - right_pad
                dst_end = frame_start + core_end_ip - left_pad
                updated_frames[:, frame_start:dst_end] = prop_imgs_sub[:, left_pad:core_end_ip]
                updated_masks[:, frame_start:dst_end] = updated_masks_sub[:, left_pad:core_end_ip]
                del prop_imgs_sub, updated_masks_sub
        else:
            prop_imgs, updated_masks = self.ppt_pipeline.img_propagation(
                masked_frames_np,
                pred_flows_bi,
                masks_dilated_np,
                shrink_memory=True,
            )
            np.multiply(prop_imgs, masks_dilated_np, out=prop_imgs)
            prop_imgs += masked_frames_np
            updated_frames = prop_imgs
        del masked_frames_np
        gc.collect()
        if debug:
            ppt_logger.info(f"Image Propagation cost: {time.time() - start_time:.4f}s")
            start_time = time.time()
        ori_frames = self._load_ori_frames_range(input_frames_dir, chunk_start, chunk_end, process_size)
        if self._use_cupy:
            ppt_logger.info("Propagation Transformer with cupy")
            self._transformer_loop_chunk_cupy(
                chunk_len=chunk_len,
                updated_frames=updated_frames,
                updated_masks=updated_masks,
                masks_dilated_np=masks_dilated_np,
                pred_flows_bi=pred_flows_bi,
                ori_frames=ori_frames,
                out_size=out_size,
                output_dir=output_dir,
                local_core_start=local_core_start,
                local_core_end=local_core_end,
                global_core_start=global_core_start,
                is_last_chunk=is_last_chunk,
            )
        else:
            ppt_logger.info("Propagation Transformer without cupy")
            self._transformer_loop_chunk_cpu(
                chunk_len=chunk_len,
                updated_frames=updated_frames,
                updated_masks=updated_masks,
                masks_dilated_np=masks_dilated_np,
                pred_flows_bi=pred_flows_bi,
                ori_frames=ori_frames,
                out_size=out_size,
                output_dir=output_dir,
                local_core_start=local_core_start,
                local_core_end=local_core_end,
                global_core_start=global_core_start,
                is_last_chunk=is_last_chunk,
            )

        if debug:
            ppt_logger.info(f"Transformer cost: {time.time() - start_time:.4f}s")

        del updated_frames, updated_masks, masks_dilated_np, pred_flows_bi, ori_frames
        gc.collect()
        if self.low_memory:
            self._trim_device_memory()

    def _get_ref_index(self, mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    def _transformer_loop_chunk_cpu(
        self,
        chunk_len,
        updated_frames,
        updated_masks,
        masks_dilated_np,
        pred_flows_bi,
        ori_frames,
        out_size,
        output_dir,
        local_core_start,
        local_core_end,
        global_core_start,
        is_last_chunk,
    ):
        comp_frames = [None] * chunk_len
        neighbor_stride = self.neighbor_length // 2 if max(updated_frames.shape[-2:]) > 540 else self.neighbor_length
        ref_num = self.subvideo_length // self.ref_stride if chunk_len > self.subvideo_length else -1
        for center in range(0, chunk_len, neighbor_stride):
            neighbor_ids = list(range(max(0, center - neighbor_stride), min(chunk_len, center + neighbor_stride + 1)))
            ref_ids = self._get_ref_index(
                center,
                neighbor_ids,
                chunk_len,
                self.ref_stride,
                ref_num,
            )
            all_ids = neighbor_ids + ref_ids
            selected_imgs = np.ascontiguousarray(updated_frames[:, all_ids])
            selected_masks = np.ascontiguousarray(masks_dilated_np[:, all_ids], dtype=np.float32)
            selected_update_masks = np.ascontiguousarray(updated_masks[:, all_ids])
            selected_pred_flows_bi = (
                np.ascontiguousarray(pred_flows_bi[0][:, neighbor_ids[:-1]]),
                np.ascontiguousarray(pred_flows_bi[1][:, neighbor_ids[:-1]]),
            )
            local_count = len(neighbor_ids)
            pred_img = self.ppt_pipeline.forward(
                selected_imgs,
                selected_pred_flows_bi[0],
                selected_pred_flows_bi[1],
                selected_masks,
                selected_update_masks,
                num_local_frames=local_count,
                shrink_memory=(is_last_chunk and center + neighbor_stride >= chunk_len),
            )
            pred_img = np.clip((pred_img + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            binary_masks = selected_masks[0, :local_count].transpose(0, 2, 3, 1)
            for local_idx, frame_idx in enumerate(neighbor_ids):
                img_pred_hwc = pred_img[0, local_idx].transpose(1, 2, 0)
                img = (img_pred_hwc * binary_masks[local_idx] + ori_frames[frame_idx] * (1.0 - binary_masks[local_idx])).astype(np.uint8)
                if comp_frames[frame_idx] is None:
                    comp_frames[frame_idx] = img
                else:
                    comp_frames[frame_idx] = (comp_frames[frame_idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5).astype(np.uint8)
        pending = deque()
        with ThreadPoolExecutor(max_workers=1) as writer:
            for local_idx in range(local_core_start, local_core_end):
                global_idx = global_core_start + (local_idx - local_core_start)
                frame = comp_frames[local_idx]
                comp_frames[local_idx] = None
                self._queue_output_frame(writer, pending, global_idx, frame, out_size, output_dir)
            while pending:
                pending.popleft().result()

    def _transformer_loop_chunk_cupy(
        self,
        chunk_len,
        updated_frames,
        updated_masks,
        masks_dilated_np,
        pred_flows_bi,
        ori_frames,
        out_size,
        output_dir,
        local_core_start,
        local_core_end,
        global_core_start,
        is_last_chunk,
    ):
        comp_frames_gpu = [None] * chunk_len
        neighbor_stride = self.neighbor_length // 2 if max(updated_frames.shape[-2:]) > 540 else self.neighbor_length
        ref_num = self.subvideo_length // self.ref_stride if chunk_len > self.subvideo_length else -1

        for center in range(0, chunk_len, neighbor_stride):
            neighbor_ids = list(range(max(0, center - neighbor_stride), min(chunk_len, center + neighbor_stride + 1)))
            ref_ids = self._get_ref_index(
                center,
                neighbor_ids,
                chunk_len,
                self.ref_stride,
                ref_num,
            )
            all_ids = neighbor_ids + ref_ids
            local_flow_ids = neighbor_ids[:-1]
            selected_imgs = cp.asarray(np.ascontiguousarray(updated_frames[:, all_ids]))
            selected_masks = cp.asarray(np.ascontiguousarray(masks_dilated_np[:, all_ids]), dtype=cp.float32)
            selected_update_masks = cp.asarray(np.ascontiguousarray(updated_masks[:, all_ids]))
            selected_flows_f = cp.asarray(np.ascontiguousarray(pred_flows_bi[0][:, local_flow_ids]))
            selected_flows_b = cp.asarray(np.ascontiguousarray(pred_flows_bi[1][:, local_flow_ids]))
            local_count = len(neighbor_ids)
            pred_img_gpu = self.ppt_pipeline.forward(
                selected_imgs,
                selected_flows_f,
                selected_flows_b,
                selected_masks,
                selected_update_masks,
                num_local_frames=local_count,
                shrink_memory=(is_last_chunk and center + neighbor_stride >= chunk_len),
            )
            pred_img_gpu = cp.clip((pred_img_gpu + 1.0) / 2.0 * 255.0, 0, 255).astype(cp.uint8)
            binary_masks = selected_masks[0, :local_count].transpose(0, 2, 3, 1)
            ori_selected = cp.asarray(ori_frames[neighbor_ids])
            for local_idx, frame_idx in enumerate(neighbor_ids):
                img_pred_hwc = pred_img_gpu[0, local_idx].transpose(1, 2, 0).astype(cp.float32)
                mask = binary_masks[local_idx]
                img = (img_pred_hwc * mask + ori_selected[local_idx] * (1.0 - mask)).astype(cp.uint8)
                if comp_frames_gpu[frame_idx] is None:
                    comp_frames_gpu[frame_idx] = img
                else:
                    comp_frames_gpu[frame_idx] = (comp_frames_gpu[frame_idx].astype(cp.float32) * 0.5 + img.astype(cp.float32) * 0.5).astype(cp.uint8)
            del (
                selected_imgs,
                selected_masks,
                selected_update_masks,
                selected_flows_f,
                selected_flows_b,
                pred_img_gpu,
                binary_masks,
                ori_selected,
            )
        pending = deque()
        with ThreadPoolExecutor(max_workers=1) as writer:
            for local_idx in range(local_core_start, local_core_end):
                global_idx = global_core_start + (local_idx - local_core_start)
                frame = cp.asnumpy(comp_frames_gpu[local_idx])
                comp_frames_gpu[local_idx] = None
                self._queue_output_frame(writer, pending, global_idx, frame, out_size, output_dir)
            while pending:
                pending.popleft().result()
