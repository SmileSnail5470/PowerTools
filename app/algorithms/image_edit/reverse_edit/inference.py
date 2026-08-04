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
        no_reconstruction: bool = False,
        vae_sample_mode: str = "sample",
        use_color_fix: bool = True,
        verbose: bool = True,
    ):
        self.model_dir = model_dir
        self.low_memory = low_memory
        self.use_io_binding = use_io_binding
        self.use_cupy = use_cupy
        self.intra_op_num_threads = intra_op_num_threads
        self.no_reconstruction = no_reconstruction
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

    def _infer_sample_size(self, image_ori, output_path, prompt, edit_prompt=None):
        inv_latent, _ = self.pipe.invert(image_ori, prompt, self.config)
        if not self.no_reconstruction:
            edit_prompt = edit_prompt if edit_prompt is not None else prompt
            rec_image = self.pipe.reconstruct(
                inv_latent,
                edit_prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=1.0,
            )[0]
            rec_image = rec_image.resize(image_ori.size, Image.Resampling.LANCZOS)
            rec_image = wavelet_color_fix(rec_image, image_ori) if self.use_color_fix and edit_prompt==prompt else rec_image
            rec_image.save(output_path)
        self.pipe.release_all()

    def infer(self, input_path, output_path, prompt, edit_prompt=None):
        image_ori = Image.open(input_path).convert("RGB")
        original_w, original_h = image_ori.size
        if original_w == self.pipe.sample_size and original_h == self.pipe.sample_size:
            self._infer_sample_size(image_ori, output_path, prompt, edit_prompt)
            return
        patch_size = self.pipe.sample_size
        stride = patch_size // 2
        
        def get_padded_size(size, p_size, st):
            if size <= p_size:
                return p_size
            remainder = (size - p_size) % st
            if remainder == 0:
                return size
            return size + (st - remainder)
            
        padded_w = get_padded_size(original_w, patch_size, stride)
        padded_h = get_padded_size(original_h, patch_size, stride)
        img_np = np.array(image_ori)
        pad_bottom = padded_h - original_h
        pad_right = padded_w - original_w
        padded_np = np.pad(img_np, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='reflect')
        padded_image = Image.fromarray(padded_np)
        if not self.no_reconstruction:
            result_canvas = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
            weight_canvas = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
            hanning_1d = np.maximum(np.hanning(patch_size), 1e-4)
            hanning_2d = np.outer(hanning_1d, hanning_1d)
            window_weight = np.expand_dims(hanning_2d, axis=-1)
            edit_prompt = edit_prompt if edit_prompt is not None else prompt
        for y in range(0, padded_h - patch_size + 1, stride):
            for x in range(0, padded_w - patch_size + 1, stride):
                patch = padded_image.crop((x, y, x + patch_size, y + patch_size))
                inv_latent, _ = self.pipe.invert(patch, prompt, self.config)
                if not self.no_reconstruction:
                    rec_patch = self.pipe.reconstruct(
                        inv_latent,
                        edit_prompt,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=1.0,
                    )[0]
                    rec_patch_np = np.array(rec_patch, dtype=np.float32)
                    result_canvas[y:y+patch_size, x:x+patch_size] += rec_patch_np * window_weight
                    weight_canvas[y:y+patch_size, x:x+patch_size] += window_weight
        if not self.no_reconstruction:
            final_img_np = result_canvas / weight_canvas
            final_img_np = np.clip(final_img_np, 0, 255).astype(np.uint8)
            rec_image = Image.fromarray(final_img_np)
            rec_image = rec_image.crop((0, 0, original_w, original_h))
            if self.use_color_fix and edit_prompt == prompt:
                rec_image = wavelet_color_fix(rec_image, image_ori)   
            rec_image.save(output_path)
        self.pipe.release_all()