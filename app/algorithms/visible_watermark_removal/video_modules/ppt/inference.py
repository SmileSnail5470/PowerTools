import gc
import os
import time
import cv2
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
        self.raft_model = None
        self.fix_flow_complete = None
        self.ppt_pipeline = None
        self._use_cupy = False

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

    def _resize_frames(self, frames, size=None):
        out_size = frames[0].size
        if size is not None:
            process_size = (size[0] - size[0] % 8, size[1] - size[1] % 8)
        else:
            process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        if out_size != process_size or size is not None:
            frames = [f.resize(process_size) for f in frames]
        return frames, process_size, out_size

    def _read_frame(self, frame_root):
        frames = []
        for fr in sorted(os.listdir(frame_root)):
            frame = cv2.imread(os.path.join(frame_root, fr))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames, None, frames[0].size

    def _binary_mask(self, mask, th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        return mask

    def _read_mask(self, mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
        masks_img = [Image.open(mpath)] if os.path.isfile(mpath) else [Image.open(os.path.join(mpath, mp)) for mp in sorted(os.listdir(mpath))]
        masks_dilated, flow_masks = [], []
        for mask_img in masks_img:
            if size is not None:
                mask_img = mask_img.resize(size, Image.NEAREST)
            mask_img = np.array(mask_img.convert("L"))
            f_mask = (
                scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
                if flow_mask_dilates > 0 else self._binary_mask(mask_img).astype(np.uint8)
            )
            flow_masks.append(Image.fromarray(f_mask * 255))
            m_mask = (
                scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
                if mask_dilates > 0 else self._binary_mask(mask_img).astype(np.uint8)
            )
            masks_dilated.append(Image.fromarray(m_mask * 255))
        if len(masks_img) == 1:
            flow_masks *= length
            masks_dilated *= length
        return flow_masks, masks_dilated

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

    def inference(self, input_frames_dir, masks_dir, output_dir, debug=True):
        try:
            return self._inference_impl(input_frames_dir, masks_dir, output_dir, debug=debug)
        finally:
            self.release()

    def _inference_impl(self, input_frames_dir, masks_dir, output_dir, debug=True):
        self.frames_name = sorted(os.listdir(input_frames_dir))
        frames, _, size = self._read_frame(frame_root=input_frames_dir)
        if self.width != -1 and self.height != -1:
            size = (self.width, self.height)
        if self.resize_ratio != 1.0:
            size = (int(self.resize_ratio * size[0]), int(self.resize_ratio * size[1]))

        frames, size, out_size = self._resize_frames(frames, size)
        os.makedirs(output_dir, exist_ok=True)

        frames_len = len(frames)
        flow_masks, masks_dilated = self._read_mask(
            masks_dir,
            frames_len,
            size,
            flow_mask_dilates=self.mask_dilation,
            mask_dilates=self.mask_dilation,
        )

        ori_frames = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames], axis=0)
        frames_np = np.ascontiguousarray(ori_frames.transpose(0, 3, 1, 2)[None]).astype(np.float32)
        frames_np *= 2.0 / 255.0
        frames_np -= 1.0
        flow_masks_np = np.ascontiguousarray((np.stack([np.asarray(mask) for mask in flow_masks], axis=0) > 0)[None, :, None]).astype(np.uint8)
        masks_dilated_np = np.ascontiguousarray((np.stack([np.asarray(mask) for mask in masks_dilated], axis=0) > 0)[None, :, None]).astype(np.uint8)
        del frames, flow_masks, masks_dilated
        gc.collect()

        video_length = frames_np.shape[1]
        self._prepare_raft()
        if debug:
            print(f"\nProcessing pipeline: [{video_length} frames] and frame shape: {frames_np.shape} and low memory: {self.low_memory} and use cupy {self._use_cupy}")
            start_time = time.time()
        raft_scale = (0.5 if min(frames_np.shape[-2], frames_np.shape[-1]) >= 480 else 1.0)
        short_clip_len = 12 if max(frames_np.shape[-2:]) <= 1280 else 6
        if video_length > short_clip_len:
            flow_shape = (
                frames_np.shape[0],
                video_length - 1,
                2,
                frames_np.shape[-2],
                frames_np.shape[-1],
            )
            gt_flows_f = np.empty(flow_shape, dtype=np.float32)
            gt_flows_b = np.empty(flow_shape, dtype=np.float32)
            flow_offset = 0
            for frame_start in range(0, video_length, short_clip_len):
                frame_end = min(video_length, frame_start + short_clip_len)
                sub_frames = frames_np[:, frame_start:frame_end] if frame_start == 0 else frames_np[:, frame_start - 1:frame_end]
                flows_f, flows_b = self.raft_model.forward(
                    sub_frames,
                    scale_factor=raft_scale,
                    shrink_memory=frame_end == video_length,
                )
                flow_count = flows_f.shape[1]
                gt_flows_f[:, flow_offset:flow_offset + flow_count] = flows_f
                gt_flows_b[:, flow_offset:flow_offset + flow_count] = flows_b
                flow_offset += flow_count
                del flows_f, flows_b
            if flow_offset != video_length - 1:
                raise RuntimeError(f"RAFT produced {flow_offset} flows for {video_length} frames")
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = self.raft_model.forward(frames_np, scale_factor=raft_scale, shrink_memory=True)
        self._release_stage("raft_model", [self.onnx_paths["raft"]])
        if debug:
            print(f"RAFT ONNX Inference Cost: {time.time() - start_time:.4f}s")
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
                core_end = slice_end - slice_start - right_pad
                core_f = sub_comb_f[:, left_pad:core_end]
                core_b = sub_comb_b[:, left_pad:core_end]
                dst_end = flow_start + core_f.shape[1]
                pred_flows_f[:, flow_start:dst_end] = core_f
                pred_flows_b[:, flow_start:dst_end] = core_b
                del sub_f, sub_b, sub_comb_f, sub_comb_b, core_f, core_b
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_f, pred_flows_b = (
                self.fix_flow_complete.forward_bidirect_flow(
                    gt_flows_bi[0],
                    gt_flows_bi[1],
                    flow_masks_np,
                    shrink_memory=True,
                )
            )
            pred_flows_bi = self.fix_flow_complete.combine_flow(
                gt_flows_bi,
                (pred_flows_f, pred_flows_b),
                flow_masks_np,
            )
            del pred_flows_f, pred_flows_b
        self._release_stage("fix_flow_complete", self._flow_complete_model_paths())
        del gt_flows_bi, flow_masks_np
        gc.collect()
        if debug:
            print(f"Flow Completion ONNX Inference Cost: {time.time() - start_time:.4f}s")
            start_time = time.time()
        for frame_start in range(0, video_length, self.subvideo_length):
            frame_end = min(video_length, frame_start + self.subvideo_length)
            np.multiply(
                frames_np[:, frame_start:frame_end],
                masks_dilated_np[:, frame_start:frame_end] == 0,
                out=frames_np[:, frame_start:frame_end],
            )
        masked_frames_np = frames_np
        frames_np = None
        self._prepare_ppt()
        subvideo_length_img_prop = min(100, self.subvideo_length)
        if video_length > subvideo_length_img_prop:
            updated_frames = np.empty_like(masked_frames_np)
            updated_masks = np.empty(masks_dilated_np.shape, dtype=np.float32)
            pad_len = 10
            for frame_start in range(0, video_length, subvideo_length_img_prop):
                slice_start = max(0, frame_start - pad_len)
                slice_end = min(video_length, frame_start + subvideo_length_img_prop + pad_len)
                left_pad = frame_start - slice_start
                right_pad = slice_end - min(video_length, frame_start + subvideo_length_img_prop)
                sub_flows_bi = (
                    pred_flows_bi[0][:, slice_start:slice_end - 1],
                    pred_flows_bi[1][:, slice_start:slice_end - 1],
                )
                prop_imgs_sub, updated_masks_sub = self.ppt_pipeline.img_propagation(
                    masked_frames_np[:, slice_start:slice_end],
                    sub_flows_bi,
                    masks_dilated_np[:, slice_start:slice_end],
                    shrink_memory=(frame_start + subvideo_length_img_prop >= video_length)
                )
                np.multiply(
                    prop_imgs_sub,
                    masks_dilated_np[:, slice_start:slice_end],
                    out=prop_imgs_sub,
                )
                prop_imgs_sub += masked_frames_np[:, slice_start:slice_end]
                core_end = slice_end - slice_start - right_pad
                dst_end = frame_start + core_end - left_pad
                updated_frames[:, frame_start:dst_end] = prop_imgs_sub[:, left_pad:core_end]
                updated_masks[:, frame_start:dst_end] = updated_masks_sub[:, left_pad:core_end]
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
            print(f"Image Propagation ONNX Inference Cost: {time.time() - start_time:.4f}s")
            start_time = time.time()

        if self._use_cupy:
            self._transformer_loop_cupy(
                video_length,
                updated_frames,
                updated_masks,
                masks_dilated_np,
                pred_flows_bi,
                ori_frames,
                out_size,
                output_dir,
            )
        else:
            self._transformer_loop_cpu(
                video_length,
                updated_frames,
                updated_masks,
                masks_dilated_np,
                pred_flows_bi,
                ori_frames,
                out_size,
                output_dir,
            )

        if debug:
            print(f"Propagation Transformer ONNX Inference Cost: {time.time() - start_time:.4f}s")
        self._release_stage("ppt_pipeline", self._flatten_model_paths(self.onnx_paths["ppt"]))
        del updated_frames, updated_masks, masks_dilated_np, pred_flows_bi
        gc.collect()

    def _transformer_loop_cpu(
        self,
        video_length,
        updated_frames,
        updated_masks,
        masks_dilated_np,
        pred_flows_bi,
        ori_frames,
        out_size,
        output_dir,
    ):
        comp_frames = [None] * video_length
        neighbor_stride = self.neighbor_length // 2 if max(updated_frames.shape[-2:]) > 540 else self.neighbor_length
        ref_num = self.subvideo_length // self.ref_stride if video_length > self.subvideo_length else -1
        next_output_idx = 0
        pending = deque()
        with ThreadPoolExecutor(max_workers=1) as writer:
            for center in range(0, video_length, neighbor_stride):
                neighbor_ids = list(range(max(0, center - neighbor_stride), min(video_length, center + neighbor_stride + 1)))
                ref_ids = self._get_ref_index(
                    center,
                    neighbor_ids,
                    video_length,
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
                    shrink_memory=(center + neighbor_stride >= video_length),
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
                while next_output_idx < center:
                    frame = comp_frames[next_output_idx]
                    comp_frames[next_output_idx] = None
                    self._queue_output_frame(
                        writer,
                        pending,
                        next_output_idx,
                        frame,
                        out_size,
                        output_dir,
                    )
                    next_output_idx += 1
            while next_output_idx < video_length:
                frame = comp_frames[next_output_idx]
                comp_frames[next_output_idx] = None
                self._queue_output_frame(
                    writer,
                    pending,
                    next_output_idx,
                    frame,
                    out_size,
                    output_dir,
                )
                next_output_idx += 1
            while pending:
                pending.popleft().result()

    def _transformer_loop_cupy(
        self,
        video_length,
        updated_frames,
        updated_masks,
        masks_dilated_np,
        pred_flows_bi,
        ori_frames,
        out_size,
        output_dir,
    ):
        comp_frames_gpu = [None] * video_length
        neighbor_stride = self.neighbor_length // 2 if max(updated_frames.shape[-2:]) > 540 else self.neighbor_length
        ref_num = self.subvideo_length // self.ref_stride if video_length > self.subvideo_length else -1
        next_output_idx = 0
        pending = deque()
        with ThreadPoolExecutor(max_workers=1) as writer:
            for center in range(0, video_length, neighbor_stride):
                neighbor_ids = list(range(max(0, center - neighbor_stride), min(video_length, center + neighbor_stride + 1)))
                ref_ids = self._get_ref_index(
                    center,
                    neighbor_ids,
                    video_length,
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
                    shrink_memory=(center + neighbor_stride >= video_length),
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
                while next_output_idx < center:
                    frame = cp.asnumpy(comp_frames_gpu[next_output_idx])
                    comp_frames_gpu[next_output_idx] = None
                    self._queue_output_frame(
                        writer,
                        pending,
                        next_output_idx,
                        frame,
                        out_size,
                        output_dir,
                    )
                    next_output_idx += 1
            while next_output_idx < video_length:
                frame = cp.asnumpy(comp_frames_gpu[next_output_idx])
                comp_frames_gpu[next_output_idx] = None
                self._queue_output_frame(
                    writer,
                    pending,
                    next_output_idx,
                    frame,
                    out_size,
                    output_dir,
                )
                next_output_idx += 1
            while pending:
                pending.popleft().result()
