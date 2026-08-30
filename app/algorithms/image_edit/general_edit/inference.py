import math
import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
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

    def _poisson_clone(self, target: Image.Image, source: Image.Image, x: int, y: int) -> Image.Image:
        target_np = np.array(target.convert("RGB"))
        source_np = np.array(source.convert("RGB"))
        target_h, target_w = target_np.shape[:2]
        h, w = source_np.shape[:2]
        if h <= 0 or w <= 0 or h > target_h or w > target_w:
            return target
        x = max(0, min(int(x), target_w - w))
        y = max(0, min(int(y), target_h - h))

        margin = 4
        pad = margin + 1
        canvas = cv2.copyMakeBorder(target_np, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        source_pad = cv2.copyMakeBorder(source_np, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        sx, sy = x, y
        region = canvas[sy:sy + source_pad.shape[0], sx:sx + source_pad.shape[1]]
        outside = np.ones(region.shape[:2], dtype=bool)
        iy0, ix0 = max(0, pad - sy), max(0, pad - sx)
        iy1 = min(region.shape[0], pad + target_h - sy)
        ix1 = min(region.shape[1], pad + target_w - sx)
        if iy1 > iy0 and ix1 > ix0:
            outside[iy0:iy1, ix0:ix1] = False
        region[outside] = source_pad[outside]
        mask = np.zeros(source_pad.shape[:2], dtype=np.uint8)
        mask[1:-1, 1:-1] = 255
        roi_w, roi_h = w + 2 * margin, h + 2 * margin
        center = (x + pad - margin + roi_w // 2, y + pad - margin + roi_h // 2)
        try:
            blended = cv2.seamlessClone(
                source_pad,
                canvas,
                mask,
                center,
                cv2.NORMAL_CLONE,
            )
        except cv2.error:
            blended = canvas
            blended[y + pad:y + pad + h, x + pad:x + pad + w] = source_np
        result = target_np.copy()
        result[y:y + h, x:x + w] = blended[y + pad:y + pad + h, x + pad:x + pad + w]
        return Image.fromarray(result)

    def _direct_paste(self, target: Image.Image, source: Image.Image, x: int, y: int) -> Image.Image:
        target_np = np.array(target.convert("RGB"))
        source_np = np.array(source.convert("RGB"))
        target_h, target_w = target_np.shape[:2]
        h, w = source_np.shape[:2]
        if h <= 0 or w <= 0 or h > target_h or w > target_w:
            return target
        x = max(0, min(int(x), target_w - w))
        y = max(0, min(int(y), target_h - h))
        result = target_np.copy()
        result[y:y + h, x:x + w] = source_np
        return Image.fromarray(result)

    def _harmonize(self, original: Image.Image, generated: Image.Image, regions: list[tuple[int, int, int, int]]) -> Image.Image:
        original = original.convert("RGB")
        generated = generated.convert("RGB")
        img_width, img_height = original.size
        if generated.size != (img_width, img_height):
            generated = generated.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
        if not regions:
            return original

        expand = 2 * max(int(self.dilate_num), 0) + 1
        blended = original
        for x1, y1, x2, y2 in regions:
            ex1 = max(0, int(x1) - expand)
            ey1 = max(0, int(y1) - expand)
            ex2 = min(img_width - 1, int(x2) + expand)
            ey2 = min(img_height - 1, int(y2) + expand)
            if ex2 <= ex1 or ey2 <= ey1:
                continue
            patch = generated.crop((ex1, ey1, ex2 + 1, ey2 + 1))
            blended = self._poisson_clone(blended, patch, ex1, ey1)
        return blended

    def _update_input_image_and_promot(self, prompt: str, image: Image.Image, task_type: str):
        if task_type == "watermark_remove":
            prompt = "Remove all logo, text, watermark, subtitle, printed text."
        return image, prompt

    def _infer_scale(self, prompt: str, image: Image.Image, regions: list[tuple[int, int, int, int]], task_type: str) -> Image.Image:
        img_width, img_height = image.size
        infer_w, infer_h, _, _ = self._resolve_infer_size([image])
        if (infer_w, infer_h) == (img_width, img_height):
            condition = image
        else:
            condition = image.resize((infer_w, infer_h), resample=Image.Resampling.LANCZOS)
        condition, prompt = self._update_input_image_and_promot(prompt=prompt, image=condition, task_type=task_type)
        result = self._infer(prompt=prompt, input_images=[condition], width=infer_w, height=infer_h)
        if result.size != (img_width, img_height):
            result = result.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)

        blended = self._harmonize(
            original=image,
            generated=result,
            regions=regions,
        )
        return blended

    @staticmethod
    def _find_mask_regions(mask: np.ndarray, min_area_ratio: float = 0.0006, min_area_abs: int = 300, max_gap: int = 8) -> list[tuple[int, int, int, int]]:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.uint8)
        h, w = mask.shape
        total_area = h * w
        binary = (mask > 127).astype(np.uint8)
        if binary.sum() == 0:
            return []
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        comps = []
        for i in range(1, num_labels):
            x, y, w0, h0, area = stats[i]
            if area < 20:
                continue
            comps.append({
                "id": i,
                "bbox": (x, y, w0, h0),
                "area": area,
                "size": max(w0, h0),
            })
        if not comps:
            return []

        def bbox_gap(a, b):
            xa, ya, wa, ha = a["bbox"]
            xb, yb, wb, hb = b["bbox"]
            gap_x = max(0, xb - (xa + wa), xa - (xb + wb))
            gap_y = max(0, yb - (ya + ha), ya - (yb + hb))
            return max(gap_x, gap_y)

        min_area = max(int(total_area * min_area_ratio), min_area_abs)
        visited = [False] * len(comps)
        clusters = []
        for i in range(len(comps)):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            cluster = [i]
            while stack:
                cur = stack.pop()
                for j in range(len(comps)):
                    if visited[j]:
                        continue
                    gap = bbox_gap(comps[cur], comps[j])
                    thresh = max(max_gap, int(0.2 * min(comps[cur]["size"], comps[j]["size"])))
                    if gap <= thresh:
                        visited[j] = True
                        stack.append(j)
                        cluster.append(j)
            cluster_area = sum(comps[idx]["area"] for idx in cluster)
            if cluster_area >= min_area:
                clusters.append(cluster)
        regions = []
        for cluster in clusters:
            x1 = min(comps[idx]["bbox"][0] for idx in cluster)
            y1 = min(comps[idx]["bbox"][1] for idx in cluster)
            x2 = max(comps[idx]["bbox"][0] + comps[idx]["bbox"][2] - 1 for idx in cluster)
            y2 = max(comps[idx]["bbox"][1] + comps[idx]["bbox"][3] - 1 for idx in cluster)
            regions.append((x1, y1, x2, y2))
        return regions

    def infer_local_patches(self, prompt, image, mask_np, task_type):
        if isinstance(mask_np, Image.Image):
            mask_np = np.array(mask_np.convert("L"))
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        img_width, img_height = image.size
        regions = self._find_mask_regions(mask_np)
        if not regions:
            raise ValueError("No valid regions found in the mask.")
        output = self._infer_scale(prompt, image, regions, task_type)
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