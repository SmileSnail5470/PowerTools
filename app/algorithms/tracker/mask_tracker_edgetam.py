import json
import logging
import os
from collections import deque
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Tuple
from app.algorithms import general_provider, general_session, general_inference_session, ORTEnvironment
ORTEnvironment.initialize()


IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

@dataclass(frozen=True)
class MaskQualityConfig:
    logit_threshold: float = 0.0
    min_mask_pixels: int = 4
    near_empty_reference_ratio: float = 0.02
    area_ratio_soft: Tuple[float, float] = (0.35, 2.8)
    area_ratio_hard: Tuple[float, float] = (0.10, 8.0)
    low_iou_threshold: float = 0.10
    centroid_shift_threshold: float = 1.50
    largest_component_ratio_min: float = 0.50
    fragmented_component_count: int = 3
    weak_topk_logit: float = 0.50
    uncertain_logit_margin: float = 0.50
    max_suspect_frames: int = 2
    area_history_size: int = 5


@dataclass(frozen=True)
class MaskQuality:
    area: int
    reference_area: float
    area_ratio: float
    previous_iou: float
    centroid_shift: float
    component_count: int
    largest_component_ratio: float
    max_logit: float
    topk_logit: float
    uncertain_ratio: float


@dataclass(frozen=True)
class MaskDecision:
    status: str
    reasons: Tuple[str, ...]


def _resize_bilinear(arr, out_hw):
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((out_hw[1], out_hw[0]), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


class MaskTrackerEdgeTAM:
    VALID = "VALID"
    SUSPECT = "SUSPECT"
    LOST = "LOST"

    def __init__(self, model_dir, quality_config: Optional[MaskQualityConfig] = None):
        providers, provider_options = general_provider()
        so = general_session()

        def _load(name):
            session = general_inference_session(
                model_path=os.path.join(model_dir, f"{name}.encmodel"),
                sess_options=so,
                providers=providers,
                provider_options=provider_options
            )
            return session

        self.image_encoder = _load("image_encoder")
        self.mask_encoder = _load("mask_encoder")
        self.memory_attention = _load("memory_attention")
        self.image_decoder = _load("image_decoder")
        self.mem_encoder = _load("mem_encoder")
        self._use_iobinding = self.image_encoder.use_cuda
        self.tpos_enc = np.load(os.path.join(model_dir, "maskmem_tpos_enc.npy"))
        with open(os.path.join(model_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.num_maskmem = self.meta["num_maskmem"]
        self.max_obj_ptrs = self.meta["max_obj_ptrs_in_encoder"]
        self.mem_dim = self.meta["mem_dim"]
        self.image_size = self.meta["image_size"]
        self.quality_config = quality_config or MaskQualityConfig()

    def _load_frame(self, path):
        img_pil = Image.open(path).convert("RGB")
        w, h = img_pil.size
        img = img_pil.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        arr = (arr - IMG_MEAN) / IMG_STD
        return arr[None].astype(np.float32), (h, w)

    def _load_mask(self, path):
        m = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        m = (m > 127).astype(np.float32)
        m_pil = Image.fromarray((m * 255).astype(np.uint8)).resize((self.image_size, self.image_size), resample=Image.NEAREST)
        m = (np.asarray(m_pil, dtype=np.float32) > 127).astype(np.float32)
        return m[None, None]

    def _build_memory(self, frame_idx, bank):
        to_cat_mem, to_cat_pos = [], []
        entries = [(0, bank[0])]
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            prev = bank.get(frame_idx - t_rel)
            if prev is not None and not prev["cond"]:
                entries.append((t_pos, prev))
        for t_pos, prev in entries:
            to_cat_mem.append(prev["mm"])
            tpos = self.tpos_enc[self.num_maskmem - t_pos - 1]
            to_cat_pos.append(prev["mmpos"] + tpos)
        spatial_memory = np.concatenate(to_cat_mem, axis=0).astype(np.float32)
        spatial_memory_pos = np.concatenate(to_cat_pos, axis=0).astype(np.float32)

        ptrs = [bank[0]["obj_ptr"]]
        for t_diff in range(1, self.max_obj_ptrs):
            t = frame_idx - t_diff
            if t < 0:
                break
            prev = bank.get(t)
            if prev is not None and not prev["cond"]:
                ptrs.append(prev["obj_ptr"])
        obj_ptrs = np.stack(ptrs, axis=0)
        num = obj_ptrs.shape[0]
        tok = obj_ptrs.reshape(num, 1, 4, self.mem_dim).transpose(0, 2, 1, 3).reshape(4 * num, 1, self.mem_dim)
        return spatial_memory, spatial_memory_pos, tok.astype(np.float32)

    def _enc_image(self, image):
        if self._use_iobinding:
            return self.image_encoder.run_with_iobinding({"image": image})
        return self.image_encoder.run(None, {"image": image})

    def _enc_mask(self, mask, pix_feat):
        feed = {"mask": mask, "pix_feat": pix_feat}
        if self._use_iobinding:
            return self.mask_encoder.run_with_iobinding_numpy(feed)[0]
        return self.mask_encoder.run(None, feed)[0]

    def _enc_mem(self, high_res_mask, pix_feat):
        feed = {"high_res_mask": high_res_mask.astype(np.float32), "pix_feat": pix_feat}
        if self._use_iobinding:
            return self.mem_encoder.run_with_iobinding_numpy(feed)
        return self.mem_encoder.run(None, feed)

    def _attend(self, feed):
        if self._use_iobinding:
            return self.memory_attention.run_with_iobinding(feed)[0]
        return self.memory_attention.run(None, feed)[0]

    def _decode(self, feed):
        if self._use_iobinding:
            return self.image_decoder.run_with_iobinding_numpy(feed)
        return self.image_decoder.run(None, feed)

    @staticmethod
    def _mask_geometry(mask: np.ndarray):
        """
        提取 Mask 的面积、连通域数量、最大连通域占比、最大连通域质心以及外接矩形对角线长度，可用于评估分割质量、判断目标是否跟丢，以及估计目标的位置和尺度变化。
        Returns:
            tuple:
                (
                    area (int),
                    component_count (int),
                    largest_component_ratio (float),
                    centroid (tuple[float, float] | None),
                    bbox_diagonal (float),
                )
                其中：
                - area: 前景像素数量。
                - component_count: 前景连通域数量（不包含背景）。
                - largest_component_ratio: 最大连通域面积占总前景面积的比例，越接近 1 表示 Mask 越完整。
                - centroid: 最大连通域质心坐标 `(x, y)`；若 Mask 为空则返回 `None`。
                - bbox_diagonal: Mask 外接矩形对角线长度，可用于目标尺度估计和位移归一化。
        """
        binary = mask.astype(np.uint8)
        area = int(np.count_nonzero(binary))
        if area == 0:
            return 0, 0, 0.0, None, 0.0
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        foreground_stats = stats[1:]
        foreground_areas = foreground_stats[:, cv2.CC_STAT_AREA]
        largest_component_ratio = float(foreground_areas.max()) / area
        largest_index = int(np.argmax(foreground_areas)) + 1
        centroid_x, centroid_y = centroids[largest_index]
        ys, xs = np.nonzero(binary)
        bbox_diagonal = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
        return area, component_count - 1, largest_component_ratio, (centroid_x, centroid_y), bbox_diagonal

    def _measure_quality(self, logits: np.ndarray, previous_mask: np.ndarray, accepted_areas: List[int]) -> MaskQuality:
        if logits.shape != previous_mask.shape:
            logits = _resize_bilinear(logits, previous_mask.shape)
        finite_logits = np.nan_to_num(logits, nan=-1e6, posinf=1e6, neginf=-1e6)
        mask = finite_logits > self.quality_config.logit_threshold
        area, component_count, largest_ratio, centroid, _ = self._mask_geometry(mask)
        previous_area, _, _, previous_centroid, previous_diagonal = self._mask_geometry(previous_mask)
        intersection = int(np.count_nonzero(mask & previous_mask))
        union = int(np.count_nonzero(mask | previous_mask))
        previous_iou = float(intersection / union) if union else 1.0
        if centroid is None or previous_centroid is None:
            centroid_shift = float("inf") if centroid != previous_centroid else 0.0
        else:
            distance = float(np.hypot(centroid[0] - previous_centroid[0], centroid[1] - previous_centroid[1]))
            centroid_shift = distance / max(previous_diagonal, 1.0)
        history = list(accepted_areas)
        reference_area = float(np.median(history)) if history else float(previous_area)
        area_ratio = float(area / reference_area) if reference_area > 0 else 0.0
        flat_logits = finite_logits.reshape(-1)
        topk_count = max(1, int(np.ceil(flat_logits.size * 0.01)))
        topk_values = np.partition(flat_logits, flat_logits.size - topk_count)[-topk_count:]
        return MaskQuality(
            area=area,
            reference_area=reference_area,
            area_ratio=area_ratio,
            previous_iou=previous_iou,
            centroid_shift=centroid_shift,
            component_count=component_count,
            largest_component_ratio=largest_ratio,
            max_logit=float(flat_logits.max()),
            topk_logit=float(topk_values.mean()),
            uncertain_ratio=float(np.mean(np.abs(flat_logits) < self.quality_config.uncertain_logit_margin)),
        )

    def _decide_quality(self, quality: MaskQuality) -> MaskDecision:
        cfg = self.quality_config
        if quality.area == 0:
            return MaskDecision(self.LOST, ("empty_mask",))
        near_empty_limit = min(
            max(int(round(quality.reference_area)), 1),
            max(cfg.min_mask_pixels, int(np.ceil(quality.reference_area * cfg.near_empty_reference_ratio))),
        )
        if quality.area < near_empty_limit:
            return MaskDecision(self.LOST, ("near_empty_mask",))
        area_soft = not (cfg.area_ratio_soft[0] <= quality.area_ratio <= cfg.area_ratio_soft[1])
        area_hard = not (cfg.area_ratio_hard[0] <= quality.area_ratio <= cfg.area_ratio_hard[1])
        low_iou = quality.previous_iou < cfg.low_iou_threshold
        centroid_jump = quality.centroid_shift > cfg.centroid_shift_threshold
        position_jump = (quality.previous_iou < 0.02 and quality.centroid_shift > 1.0)
        if position_jump:
            return MaskDecision(self.LOST, ("position_jump",))
        fragmented = (
            quality.component_count > cfg.fragmented_component_count
            and quality.largest_component_ratio < cfg.largest_component_ratio_min
        )
        weak_logits = quality.topk_logit < cfg.weak_topk_logit
        reasons = []
        if area_soft:
            reasons.append("area_change")
        if low_iou:
            reasons.append("low_iou")
        if centroid_jump:
            reasons.append("centroid_jump")
        if fragmented:
            reasons.append("fragmented")
        if weak_logits:
            reasons.append("weak_logits")
        if area_hard and (weak_logits or centroid_jump or fragmented):
            return MaskDecision(self.LOST, tuple(reasons))
        if weak_logits and (low_iou or fragmented):
            return MaskDecision(self.SUSPECT, tuple(reasons))
        if area_soft and (low_iou or centroid_jump or fragmented):
            return MaskDecision(self.SUSPECT, tuple(reasons))
        if low_iou and (centroid_jump or fragmented):
            return MaskDecision(self.SUSPECT, tuple(reasons))
        return MaskDecision(self.VALID, ())

    @staticmethod
    def _quality_log(frame_idx: int, decision: MaskDecision, quality: MaskQuality):
        logging.getLogger("subprocess").warning(
            "Tracker frame=%d status=%s reasons=%s area=%d area_ratio=%.3f "
            "iou=%.3f centroid_shift=%.3f components=%d largest_component=%.3f "
            "max_logit=%.3f topk_logit=%.3f uncertain_ratio=%.3f",
            frame_idx,
            decision.status,
            ",".join(decision.reasons),
            quality.area,
            quality.area_ratio,
            quality.previous_iou,
            quality.centroid_shift,
            quality.component_count,
            quality.largest_component_ratio,
            quality.max_logit,
            quality.topk_logit,
            quality.uncertain_ratio,
        )

    def _prune_bank(self, bank, current_frame_idx):
        keep_window = max(self.num_maskmem, self.max_obj_ptrs) + 3
        min_keep = current_frame_idx - keep_window
        keys_to_remove = [k for k in bank if k != 0 and k < min_keep]
        for k in keys_to_remove:
            del bank[k]

    def inference(self, mask_path, frames_dir, reverse: bool = False, output_dir: Optional[str] = None):
        frames_list = sorted(
            (
                os.path.join(frames_dir, item)
                for item in os.listdir(frames_dir)
                if os.path.isfile(os.path.join(frames_dir, item))
            ),
            reverse=reverse,
        )
        assert len(frames_list) >= 1
        bank = {}
        mask_out = output_dir or os.path.dirname(mask_path)
        os.makedirs(mask_out, exist_ok=True)
        image, (height0, width0) = self._load_frame(frames_list[0])
        mask = self._load_mask(mask_path)
        initial_mask = mask[0, 0].astype(bool)
        initial_area = int(np.count_nonzero(initial_mask))
        if initial_area == 0:
            raise ValueError("Tracker reference mask must not be empty")
        pix_feat, _, _, _, _ = self._enc_image(image)
        obj_ptr = self._enc_mask(mask, pix_feat)
        high_res = mask * 20.0 - 10.0
        mm, mmpos = self._enc_mem(high_res, pix_feat)
        bank[0] = {"mm": mm, "mmpos": mmpos, "obj_ptr": obj_ptr, "cond": True}
        del image, pix_feat, high_res
        Image.fromarray(
            (self._mask_to_orig(mask[0, 0], (height0, width0)) * 255).astype(np.uint8)
        ).save(os.path.join(mask_out, f"{os.path.splitext(os.path.basename(frames_list[0]))[0]}.png"))

        previous_reliable_mask = initial_mask
        accepted_areas = deque([initial_area], maxlen=self.quality_config.area_history_size)
        pending_start_idx = None
        suspect_count = 0

        for frame_idx in range(1, len(frames_list)):
            image, (height, width) = self._load_frame(frames_list[frame_idx])
            pix_feat, hr0, hr1, vfeat, vpos = self._enc_image(image)
            del image
            spatial_memory, spatial_pos, pointer_tokens = self._build_memory(frame_idx, bank)
            image_embed = self._attend({
                "vision_feat": vfeat,
                "vision_pos_embed": vpos,
                "spatial_memory": spatial_memory,
                "spatial_memory_pos": spatial_pos,
                "obj_ptr_tokens": pointer_tokens,
            })
            del vfeat, vpos, spatial_memory, spatial_pos, pointer_tokens
            obj_ptr, high_res = self._decode({
                "image_embed": image_embed,
                "high_res_feat0": hr0,
                "high_res_feat1": hr1,
            })
            del image_embed, hr0, hr1
            raw_logits = high_res[0, 0]
            quality = self._measure_quality(raw_logits, previous_reliable_mask, accepted_areas)
            decision = self._decide_quality(quality)
            if decision.status != self.VALID:
                self._quality_log(frame_idx, decision, quality)
            if decision.status == self.LOST:
                return pending_start_idx if pending_start_idx is not None else frame_idx

            output_logits = _resize_bilinear(raw_logits, (height, width))
            output_mask = output_logits > self.quality_config.logit_threshold
            del output_logits
            output_path = os.path.join(mask_out, f"{os.path.splitext(os.path.basename(frames_list[frame_idx]))[0]}.png")
            Image.fromarray((output_mask * 255).astype(np.uint8)).save(output_path)
            del output_mask
            if decision.status == self.SUSPECT:
                if pending_start_idx is None:
                    pending_start_idx = frame_idx
                    suspect_count = 1
                else:
                    suspect_count += 1
                del pix_feat, high_res, obj_ptr
                if suspect_count >= self.quality_config.max_suspect_frames:
                    return pending_start_idx
                continue
            pending_start_idx = None
            suspect_count = 0
            reliable_logits = raw_logits
            if reliable_logits.shape != previous_reliable_mask.shape:
                reliable_logits = _resize_bilinear(reliable_logits, previous_reliable_mask.shape)
            previous_reliable_mask = reliable_logits > self.quality_config.logit_threshold
            accepted_areas.append(int(np.count_nonzero(previous_reliable_mask)))
            mm, mmpos = self._enc_mem(high_res, pix_feat)
            del high_res, pix_feat
            bank[frame_idx] = {"mm": mm, "mmpos": mmpos, "obj_ptr": obj_ptr, "cond": False}
            self._prune_bank(bank, frame_idx)
        return -1

    def _mask_to_orig(self, mask_s, out_hw):
        mask = _resize_bilinear(mask_s.astype(np.float32), out_hw)
        return mask > 0.5
