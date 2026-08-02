from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from app.algorithms.image_edit.color_fix import wavelet_color_fix
from app.algorithms.image_edit.general_edit.pipeline import Pipeline

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class ImageEditInference:
    def __init__(
        self,
        model_dir,
        low_memory: bool = True,
        use_io_binding: bool | None = None,
        use_cupy: bool = True,
        intra_op_num_threads: int | None = None,
        num_inference_steps: int = 4,
        use_color_fix: bool = True,
        dilate_num: int = 4,
        verbose: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        self.use_color_fix = use_color_fix
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
        if image_list is None or len(image_list) == 0:
            return 1024, 1024, None, None
        img = image_list[0]
        img_width, img_height = img.size
        if max(img_width, img_height) <= 1024:
            new_height = img_height
            new_width = img_width
        else:
            aspect_ratio = img_width / img_height
            if aspect_ratio >= 1:
                new_width = 1024
                new_height = int(1024 / aspect_ratio)
            else:
                new_height = 1024
                new_width = int(1024 * aspect_ratio)
        new_width = round(new_width / 8) * 8
        new_height = round(new_height / 8) * 8
        new_width = max(256, min(1024, new_width))
        new_height = max(256, min(1024, new_height))
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

    def infer_local_patches(self, prompt, output_path, image, mask_np, patch_size=512):
        img_np = np.array(image)
        H, W, C = img_np.shape
        if len(mask_np.shape) > 2:
            mask_np = mask_np[:, :, 0]
        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        processed_mask = np.zeros_like(binary_mask)
        result_img_np = img_np.copy()
        for i in range(1, num_labels):
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            if processed_mask[cy, cx] == 255:
                continue
            target_x = stats[i, cv2.CC_STAT_LEFT]
            target_y = stats[i, cv2.CC_STAT_TOP]
            target_w = stats[i, cv2.CC_STAT_WIDTH]
            target_h = stats[i, cv2.CC_STAT_HEIGHT]
            if target_w > patch_size or target_h > patch_size:
                x1 = max(0, cx - patch_size // 2)
                y1 = max(0, cy - patch_size // 2)
            else:
                box_cx = target_x + target_w // 2
                box_cy = target_y + target_h // 2
                x1 = max(0, box_cx - patch_size // 2)
                y1 = max(0, box_cy - patch_size // 2)
            x2 = min(W, x1 + patch_size)
            y2 = min(H, y1 + patch_size)
            if x2 - x1 < patch_size and W >= patch_size:
                x1, x2 = (W - patch_size, W) if x1 > 0 else (0, patch_size)
            if y2 - y1 < patch_size and H >= patch_size:
                y1, y2 = (H - patch_size, H) if y1 > 0 else (0, patch_size)
            crop_w = (x2 - x1) // 8 * 8
            crop_h = (y2 - y1) // 8 * 8
            x2, y2 = x1 + crop_w, y1 + crop_h
            if crop_w < 256 or crop_h < 256:
                continue
            img_patch = result_img_np[y1:y2, x1:x2]
            local_labels = labels[y1:y2, x1:x2]
            unique_labels_in_patch = np.unique(local_labels)
            valid_mask_in_patch = np.zeros_like(local_labels, dtype=np.uint8)
            for lbl in unique_labels_in_patch:
                if lbl == 0:
                    continue
                lbl_x = stats[lbl, cv2.CC_STAT_LEFT]
                lbl_y = stats[lbl, cv2.CC_STAT_TOP]
                lbl_w = stats[lbl, cv2.CC_STAT_WIDTH]
                lbl_h = stats[lbl, cv2.CC_STAT_HEIGHT]
                is_fully_contained = (
                    lbl_x >= x1 and 
                    lbl_y >= y1 and 
                    (lbl_x + lbl_w) <= x2 and 
                    (lbl_y + lbl_h) <= y2
                )
                if is_fully_contained or lbl == i:
                    valid_mask_in_patch[local_labels == lbl] = 255
            if not np.any(valid_mask_in_patch):
                continue
            img_patch_pil = Image.fromarray(img_patch)
            inferred_patch_pil = self._infer(
                prompt=prompt, 
                input_images=[img_patch_pil], 
                width=crop_w, 
                height=crop_h
            )
            inferred_patch_np = np.array(inferred_patch_pil)
            if self.dilate_num > 0:
                kernel = np.ones((3, 3), np.uint8)
                local_mask_process = cv2.dilate(valid_mask_in_patch, kernel, iterations=self.dilate_num)
            else:
                local_mask_process = valid_mask_in_patch
            local_mask_bool = local_mask_process > 127
            result_img_np[y1:y2, x1:x2][local_mask_bool] = inferred_patch_np[local_mask_bool]
            processed_mask[y1:y2, x1:x2][valid_mask_in_patch > 127] = 255
        final_image = Image.fromarray(result_img_np)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_image.save(output_path)

    def infer(self, prompt, output_path, input_path = None, mask = None):
        if input_path is not None:
            image_list = [Image.open(p).convert("RGB") for p in self._collect(input_path)]
        else:
            image_list = None
        if mask is not None and image_list is None:
            raise ValueError("Mask is provided but no input images are available.")
        if mask is None:
            width, height, ori_width, ori_height = self._update_dimensions_from_image(image_list)
            image = self._infer(prompt=prompt, input_images=image_list, width=width, height=height)
            image = wavelet_color_fix(image, image_list[0]) if image_list is not None and self.use_color_fix else image
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if ori_height and ori_width:
                image = image.resize((ori_width, ori_height), resample=Image.Resampling.LANCZOS)
            image.save(output_path)
        else:
            self.infer_local_patches(prompt, output_path, image_list[0], mask)