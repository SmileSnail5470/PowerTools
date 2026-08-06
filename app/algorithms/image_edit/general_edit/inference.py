import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from app.algorithms.image_edit.color_fix import calibrate_color_fix
from app.algorithms.image_edit.general_edit.pipeline import Pipeline

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
CROP_EXPAND_PIXELS = 120
SCALE_THRESHOLD = 1408 if int(os.environ["POWERTOOLS_GPU_MEMORY_LIMIT"]) >= 12 else 1152
FEATHER_RADIUS = 2


class ImageEditInference:
    def __init__(
        self,
        model_dir,
        low_memory: bool = True,
        use_io_binding: bool | None = None,
        use_cupy: bool = True,
        intra_op_num_threads: int | None = None,
        num_inference_steps: int = 4,
        dilate_num: int = 4,
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

    def _update_dimensions_from_image(self, image_list):
        gpu_memory_limit = int(os.environ.get("POWERTOOLS_GPU_MEMORY_LIMIT", 16))
        if image_list is None or len(image_list) == 0:
            res_size = 1024 if gpu_memory_limit >= 12 else 960 if gpu_memory_limit >= 8 else 768
            return res_size, res_size, None, None
        max_side = 1024 if gpu_memory_limit >= 12 else 832
        img = image_list[0]
        img_width, img_height = img.size
        if max(img_width, img_height) <= max_side:
            new_height = img_height
            new_width = img_width
        else:
            aspect_ratio = img_width / img_height
            if aspect_ratio >= 1:
                new_width = max_side
                new_height = int(max_side / aspect_ratio)
            else:
                new_height = max_side
                new_width = int(max_side * aspect_ratio)
        new_width = round(new_width / 8) * 8
        new_height = round(new_height / 8) * 8
        new_width = max(256, min(max_side, new_width))
        new_height = max(256, min(max_side, new_height))
        return new_width, new_height, img_width, img_height

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
        num_labels, labels = cv2.connectedComponents(binary)
        regions = []
        for label_id in range(1, num_labels):
            ys, xs = np.where(labels == label_id)
            if len(xs) == 0:
                continue
            regions.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
        return regions

    @staticmethod
    def _compute_crop_box(bbox: tuple[int, int, int, int], img_width: int, img_height: int, expand: int = CROP_EXPAND_PIXELS) -> tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = bbox
        x1 = max(0, x_min - expand)
        y1 = max(0, y_min - expand)
        x2 = min(img_width, x_max + expand)
        y2 = min(img_height, y_max + expand)
        x1 = (x1 // 8) * 8
        y1 = (y1 // 8) * 8
        x2 = min(img_width, ((x2 + 7) // 8) * 8)
        y2 = min(img_height, ((y2 + 7) // 8) * 8)
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 256:
            deficit = 256 - crop_w
            x1 = max(0, x1 - deficit // 2)
            x2 = min(img_width, x1 + 256)
            x1 = max(0, x2 - 256)
        if crop_h < 256:
            deficit = 256 - crop_h
            y1 = max(0, y1 - deficit // 2)
            y2 = min(img_height, y1 + 256)
            y1 = max(0, y2 - 256)
        return x1, y1, x2, y2

    def _dilate_mask(self, mask_np: np.ndarray) -> np.ndarray:
        if self.dilate_num <= 0:
            return mask_np
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.dilate_num + 1, 2 * self.dilate_num + 1))
        return cv2.dilate(mask_np, kernel, iterations=1)

    @staticmethod
    def _feather_mask(mask_np: np.ndarray, radius: int = FEATHER_RADIUS) -> np.ndarray:
        if radius <= 0:
            return (mask_np > 127).astype(np.float32)
        ksize = radius * 2 + 1
        blurred = cv2.GaussianBlur(mask_np.astype(np.float32), (ksize, ksize), 0)
        blurred = blurred / 255.0
        return np.clip(blurred, 0.0, 1.0)

    @staticmethod
    def _blend_with_mask(original_np: np.ndarray, generated_np: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if alpha.ndim == 2:
            alpha = alpha[:, :, np.newaxis]
        original = original_np.astype(np.float32)
        generated = generated_np.astype(np.float32)
        blended = original * (1.0 - alpha) + generated * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _infer_crop_region(self, prompt: str, image: Image.Image, mask_np: np.ndarray, crop_box: tuple[int, int, int, int]) -> Image.Image:
        x1, y1, x2, y2 = crop_box
        crop_w, crop_h = x2 - x1, y2 - y1
        cropped_image = image.crop((x1, y1, x2, y2))
        dilated_mask = self._dilate_mask(mask_np)
        crop_mask = dilated_mask[y1:y2, x1:x2]
        infer_w, infer_h, _, _ = self._update_dimensions_from_image([cropped_image])
        condition_resized = cropped_image.resize((infer_w, infer_h), resample=Image.Resampling.LANCZOS)
        result = self._infer(
            prompt=prompt,
            input_images=[condition_resized],
            width=infer_w,
            height=infer_h,
        )
        result_resized = result.resize((crop_w, crop_h), resample=Image.Resampling.LANCZOS)
        cropped_np = np.array(cropped_image)
        result_np = np.array(result_resized)
        fixed_result = calibrate_color_fix(
            original_bgr=cropped_np,
            generated_bgr=result_np,
            mask_gray=crop_mask,
        )
        alpha = self._feather_mask(crop_mask, radius=FEATHER_RADIUS)
        blended_crop = self._blend_with_mask(cropped_np, fixed_result, alpha)
        output = image.copy()
        output_np = np.array(output)
        output_np[y1:y2, x1:x2] = blended_crop
        return Image.fromarray(output_np)

    def _infer_scale(self, prompt: str, image: Image.Image, mask_np: np.ndarray) -> Image.Image:
        img_width, img_height = image.size
        image_np = np.array(image)
        dilated_mask = self._dilate_mask(mask_np)
        infer_w, infer_h, _, _ = self._update_dimensions_from_image([image])
        condition_resized = image.resize((infer_w, infer_h), resample=Image.Resampling.LANCZOS)
        result = self._infer(
            prompt=prompt,
            input_images=[condition_resized],
            width=infer_w,
            height=infer_h,
        )
        result_resized = result.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
        result_np = np.array(result_resized)
        fixed_result = calibrate_color_fix(
            original_bgr=image_np,
            generated_bgr=result_np,
            mask_gray=dilated_mask,
        )
        alpha = self._feather_mask(dilated_mask, radius=FEATHER_RADIUS)
        blended = self._blend_with_mask(image_np, fixed_result, alpha)
        return Image.fromarray(blended)

    def infer_local_patches(self, prompt, image, mask_np):
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
            output = self._infer_scale(prompt, image, mask_np)
        else:
            output = image.copy()
            for region_bbox in regions:
                crop_box = self._compute_crop_box(region_bbox, img_width, img_height)
                output = self._infer_crop_region(prompt, output, mask_np, crop_box)
        if output.size != (img_width, img_height):
            output = output.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
        return np.array(output)

    def infer(self, prompt, input_path=None, mask=None):
        if input_path is not None:
            image_list = [Image.open(p).convert("RGB") for p in self._collect(input_path)]
        else:
            image_list = None
        if mask is not None and image_list is None:
            raise ValueError("Mask is provided but no input images are available.")
        if mask is None:
            width, height, ori_width, ori_height = self._update_dimensions_from_image(image_list)
            image = self._infer(prompt=prompt, input_images=image_list, width=width, height=height)
            if ori_height and ori_width:
                image = image.resize((ori_width, ori_height), resample=Image.Resampling.LANCZOS)
            return np.array(image)
        else:
            return self.infer_local_patches(prompt, image_list[0], mask)