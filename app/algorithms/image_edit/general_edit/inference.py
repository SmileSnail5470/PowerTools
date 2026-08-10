import math
import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from app.algorithms.image_edit.color_fix import lab_color_fix
from app.algorithms.image_edit.general_edit.pipeline import Pipeline

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
CROP_EXPAND_PIXELS = 120
MIN_CROP_SIDE = 256
# pipeline 侧的长宽比校验上限是 8:1，留一点余量
MAX_ASPECT_RATIO = 7.5


class ImageEditInference:
    def __init__(
        self,
        model_dir,
        low_memory: bool = True,
        use_io_binding: bool | None = None,
        use_cupy: bool = True,
        intra_op_num_threads: int | None = None,
        num_inference_steps: int = 4,
        dilate_num: int = 2,
        verbose: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        self.dilate_num = dilate_num
        self.pipe = Pipeline.from_pretrained(
            model_dir,
            low_memory=low_memory,
            use_io_binding=use_io_binding,
            use_cupy=use_cupy,
            intra_op_num_threads=intra_op_num_threads,
            verbose=verbose,
        )

    @staticmethod
    def _collect(input_path) -> list[str]:
        path = Path(input_path)
        if path.is_dir():
            return sorted(str(p) for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        return [str(path)]

    @property
    def _align(self) -> int:
        return max(8, int(self.pipe.vae_scale_factor) * 2)

    def _size_budget(self) -> tuple[int, int]:
        gpu_memory_limit = int(os.environ.get("POWERTOOLS_GPU_MEMORY_LIMIT", 16))
        if gpu_memory_limit >= 16:
            area, max_side = 1024 * 1024, 1536
        elif gpu_memory_limit >= 12:
            area, max_side = 832 * 832, 1280
        else:
            area, max_side = 768 * 768, 1024
        return min(area, int(self.pipe.max_condition_area)), max_side

    def _compute_infer_size(self, width: int, height: int) -> tuple[int, int]:
        align = self._align
        min_side = max(64, align)
        area, max_side = self._size_budget()
        if (
            width % align == 0
            and height % align == 0
            and width >= min_side
            and height >= min_side
            and width * height <= area
            and max(width, height) <= max_side
        ):
            return width, height
        scale = min(1.0, math.sqrt(area / float(max(width * height, 1))), max_side / float(max(width, height, 1)))
        target_w = max(min_side, int(round(width * scale / align)) * align)
        target_h = max(min_side, int(round(height * scale / align)) * align)
        while target_w * target_h > area and (target_w > min_side or target_h > min_side):
            if target_w >= target_h and target_w > min_side:
                target_w -= align
            elif target_h > min_side:
                target_h -= align
            else:
                break
        return target_w, target_h

    def _resolve_infer_size(self, image_list):
        if not image_list:
            area, _ = self._size_budget()
            side = max(self._align, (int(math.sqrt(area)) // self._align) * self._align)
            return side, side, None, None
        img_width, img_height = image_list[0].size
        infer_w, infer_h = self._compute_infer_size(img_width, img_height)
        return infer_w, infer_h, img_width, img_height

    def _infer(self, prompt, input_images=None, seed=42, width=1024, height=1024, num_inference_steps=None):
        """
            input_images = None 表示文生图模式
        """
        output = self.pipe(
            prompt=prompt,
            image=list(input_images) if input_images else None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps or self.num_inference_steps,
            seed=seed,
        )
        self.last_timings = output.timings
        return output.images[0]

    @staticmethod
    def _find_mask_regions(mask_np: np.ndarray) -> list[tuple[int, int, int, int]]:
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        binary = (mask_np > 127).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        regions = []
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            if area <= 0 or w <= 0 or h <= 0:
                continue
            regions.append((int(x), int(y), int(x + w - 1), int(y + h - 1)))
        return regions

    def _align_span(
        self,
        lo: float,
        hi: float,
        limit: int,
        min_size: int,
        must_lo: float | None = None,
        must_hi: float | None = None,
    ) -> tuple[int, int]:
        align = self._align
        limit = int(limit)
        max_size = (limit // align) * align
        if max_size < align:
            return 0, limit
        need_lo = lo if must_lo is None else must_lo
        need_hi = hi if must_hi is None else must_hi
        need_lo = max(0.0, min(float(need_lo), float(limit)))
        need_hi = max(0.0, min(float(need_hi), float(limit)))
        if math.ceil(need_hi) - math.floor(need_lo) > max_size:
            return 0, limit
        size = max(int(min_size), int(math.ceil(hi - lo)))
        size = ((size + align - 1) // align) * align
        size = min(size, max_size)
        center = (lo + hi) * 0.5
        start = int(round(center - size / 2.0))
        start = min(start, int(math.floor(need_lo)))
        start = max(start, int(math.ceil(need_hi)) - size)
        start = max(0, min(start, limit - size))
        if start > need_lo or start + size < need_hi:
            return 0, limit
        return start, start + size

    def _compute_crop_box(
        self,
        bbox: tuple[int, int, int, int],
        img_width: int,
        img_height: int,
        expand: int = CROP_EXPAND_PIXELS,
    ) -> tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = bbox
        x1, x2 = self._align_span(x_min - expand, x_max + 1 + expand, img_width, MIN_CROP_SIDE, x_min, x_max + 1)
        y1, y2 = self._align_span(y_min - expand, y_max + 1 + expand, img_height, MIN_CROP_SIDE, y_min, y_max + 1)
        for _ in range(8):
            crop_w, crop_h = x2 - x1, y2 - y1
            if max(crop_w / crop_h, crop_h / crop_w) <= MAX_ASPECT_RATIO:
                break
            if crop_w > crop_h:
                need = min(img_height, int(math.ceil(crop_w / MAX_ASPECT_RATIO)))
                y1, y2 = self._align_span(y1, y2, img_height, need, y_min, y_max + 1)
            else:
                need = min(img_width, int(math.ceil(crop_h / MAX_ASPECT_RATIO)))
                x1, x2 = self._align_span(x1, x2, img_width, need, x_min, x_max + 1)
            if (x2 - x1, y2 - y1) == (crop_w, crop_h):
                break
        return x1, y1, x2, y2

    def _dilate_mask(self, mask_np: np.ndarray, radius: int | None = None) -> np.ndarray:
        radius = self.dilate_num if radius is None else radius
        if radius <= 0:
            return mask_np
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(mask_np, kernel, iterations=1)

    def _poisson_clone(self, target: Image.Image, source: Image.Image, x: int, y: int) -> Image.Image:
        target_np = np.array(target.convert("RGB"))
        source_np = np.array(source.convert("RGB"))
        h, w = source_np.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        center = (x + w // 2, y + h // 2)
        result = cv2.seamlessClone(
            source_np,
            target_np,
            mask,
            center,
            cv2.NORMAL_CLONE,
        )
        return Image.fromarray(result)

    def _harmonize(self, original: Image.Image, generated: Image.Image, blend_mask: np.ndarray) -> np.ndarray:
        blended = lab_color_fix(
            original_img=original,
            generated_img=generated,
            mask_gray=blend_mask
        )
        return blended

    def _update_input_image_and_promot(self, prompt: str, image: Image.Image, mask_np: np.ndarray, task_type: str):
        if task_type == "watermark_remove":
            img_np = np.array(image, dtype=np.float32)
            overlay = img_np.copy()
            color = np.array([0, 0, 255], dtype=np.float32)
            mask_bool = mask_np > 0
            overlay[mask_bool] = overlay[mask_bool] * 0.55 + color * 0.45
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            preprocess_image = Image.fromarray(overlay, mode="RGB")
            prompt = "Remove the highlighted blue area."
            return preprocess_image, prompt
        return image, prompt

    def _infer_crop_region(self, prompt: str, image: Image.Image, mask_np: np.ndarray, crop_box: tuple[int, int, int, int], task_type: str) -> Image.Image:
        x1, y1, x2, y2 = crop_box
        crop_w, crop_h = x2 - x1, y2 - y1
        cropped_image = image.crop((x1, y1, x2, y2))
        infer_w, infer_h = self._compute_infer_size(crop_w, crop_h)
        blend_mask = self._dilate_mask(mask_np)[y1:y2, x1:x2]
        if (infer_w, infer_h) == (crop_w, crop_h):
            condition = cropped_image
        else:
            condition = cropped_image.resize((infer_w, infer_h), resample=Image.Resampling.LANCZOS)
        condition, prompt = self._update_input_image_and_promot(prompt=prompt, image=condition, mask_np=blend_mask, task_type=task_type)
        result = self._infer(prompt=prompt, input_images=[condition], width=infer_w, height=infer_h)
        if result.size != (crop_w, crop_h):
            result = result.resize((crop_w, crop_h), resample=Image.Resampling.LANCZOS)

        blended_crop = self._harmonize(
            original=cropped_image,
            generated=result,
            blend_mask=blend_mask,
        )
        output = self._poisson_clone(target=image, source=blended_crop, x=x1, y=y1)
        return output

    def _infer_scale(self, prompt: str, image: Image.Image, mask_np: np.ndarray, task_type: str) -> Image.Image:
        img_width, img_height = image.size
        infer_w, infer_h, _, _ = self._resolve_infer_size([image])
        blend_mask = self._dilate_mask(mask_np)
        if (infer_w, infer_h) == (img_width, img_height):
            condition = image
        else:
            condition = image.resize((infer_w, infer_h), resample=Image.Resampling.LANCZOS)
        condition, prompt = self._update_input_image_and_promot(prompt=prompt, image=condition, mask_np=blend_mask, task_type=task_type)
        result = self._infer(prompt=prompt, input_images=[condition], width=infer_w, height=infer_h)
        if result.size != (img_width, img_height):
            result = result.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)

        blended = self._harmonize(
            original=image,
            generated=result,
            blend_mask=blend_mask,
        )
        return blended

    def infer_local_patches(self, prompt, image, mask_np, task_type):
        if isinstance(mask_np, Image.Image):
            mask_np = np.array(mask_np.convert("L"))
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        mask_np = mask_np.astype(np.uint8)
        img_width, img_height = image.size
        regions = self._find_mask_regions(mask_np)
        if not regions:
            raise ValueError("No valid regions found in the mask.")
        num_regions = len(regions)
        if num_regions >= 16:
            output = self._infer_scale(prompt, image, mask_np, task_type)
        else:
            output = image.copy()
            for region_bbox in regions:
                crop_box = self._compute_crop_box(region_bbox, img_width, img_height)
                output = self._infer_crop_region(prompt, output, mask_np, crop_box, task_type)
        if output.size != (img_width, img_height):
            output = output.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
        return np.array(output)

    def infer(self, prompt=None, input_path=None, mask=None, task_type="general_edit"):
        if input_path is not None:
            image_list = [Image.open(p).convert("RGB") for p in self._collect(input_path)]
        else:
            image_list = None
        if mask is not None and image_list is None:
            raise ValueError("Mask is provided but no input images are available.")
        if mask is None:
            if prompt is None:
                raise ValueError("Mask is not provided but no promot are available")
            width, height, ori_width, ori_height = self._resolve_infer_size(image_list)
            image = self._infer(prompt=prompt, input_images=image_list, width=width, height=height)
            if ori_height and ori_width and image.size != (ori_width, ori_height):
                image = image.resize((ori_width, ori_height), resample=Image.Resampling.LANCZOS)
            return np.array(image)
        else:
            return self.infer_local_patches(prompt, image_list[0], mask, task_type)