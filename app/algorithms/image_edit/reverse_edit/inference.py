import numpy as np
from PIL import Image
from app.algorithms.image_edit.color_fix import wavelet_color_fix
from app.algorithms.image_edit.reverse_edit.utils import RunConfig
from app.algorithms.image_edit.reverse_edit.pipeline import ORTPipeline


class ReverseEditInference:
    def __init__(
        self,
        model_dir,
        low_memory: bool = True,
        use_io_binding: bool | None = None,
        use_cupy: bool = True,
        intra_op_num_threads: int | None = None,
        vae_sample_mode: str = "sample",
        use_color_fix: bool = True,
        verbose: bool = True,
    ):
        self.model_dir = model_dir
        self.low_memory = low_memory
        self.use_io_binding = use_io_binding
        self.use_cupy = use_cupy
        self.intra_op_num_threads = intra_op_num_threads
        self.vae_sample_mode = vae_sample_mode
        self.use_color_fix = use_color_fix
        self.verbose = verbose
        self.pipe = ORTPipeline(
            self.model_dir,
            use_io_binding=self.use_io_binding,
            use_cupy=self.use_cupy,
            low_memory=self.low_memory or None,
            intra_op_num_threads=self.intra_op_num_threads or None,
            vae_sample_mode=self.vae_sample_mode,
        )
        self.config = RunConfig()

    @staticmethod
    def _normalize_region(region, image_size):
        if region is None:
            return None
        if isinstance(region, dict):
            values = (region.get("x"), region.get("y"), region.get("w"), region.get("h"))
        else:
            try:
                values = tuple(region)
            except TypeError as exc:
                raise ValueError("region must be a dict or a four-value sequence") from exc
            if len(values) != 4:
                raise ValueError("region must contain x, y, w and h")
        if any(value is None for value in values):
            raise ValueError("region must contain x, y, w and h")
        x, y, width, height = (int(value) for value in values)
        image_width, image_height = image_size
        x1 = max(0, min(x, image_width))
        y1 = max(0, min(y, image_height))
        x2 = max(x1, min(x + width, image_width))
        y2 = max(y1, min(y + height, image_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("region must overlap the input image")
        return x1, y1, x2, y2

    def _infer_image(self, image, prompt, edit_prompt):
        original_w, original_h = image.size
        target_size = self.pipe.sample_size
        ratio = target_size / max(original_w, original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)
        resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        pad_bottom = target_size - new_h
        pad_right = target_size - new_w
        img_np = np.array(resized_img)
        padded_np = np.pad(img_np, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='reflect')
        padded_image = Image.fromarray(padded_np)
        inv_latent, _ = self.pipe.invert(padded_image, prompt, self.config)
        rec_image = self.pipe.reconstruct(
            inv_latent,
            edit_prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=1.0,
        )[0]
        rec_image = rec_image.crop((0, 0, new_w, new_h))
        if rec_image.size != image.size:
            rec_image = rec_image.resize(image.size, Image.Resampling.LANCZOS)
        return rec_image

    def infer(
        self,
        input_path,
        output_path,
        prompt,
        edit_prompt=None,
        region=None,
        high_quality_output=False,
    ):
        image_ori = Image.open(input_path).convert("RGB")
        edit_prompt = edit_prompt if edit_prompt is not None else prompt
        normalized_region = self._normalize_region(region, image_ori.size)
        processed_image = self._infer_image(image_ori, prompt, edit_prompt)
        if normalized_region is None:
            output_image = processed_image
        else:
            x1, y1, x2, y2 = normalized_region
            output_image = processed_image.copy()
            output_image.paste(image_ori.crop((x1, y1, x2, y2)), (x1, y1))
        # 处理图片超分
        pass
        if self.use_color_fix and edit_prompt == prompt:
            output_image = wavelet_color_fix(output_image, image_ori)
        output_image.save(output_path)
        self.pipe.release_all()