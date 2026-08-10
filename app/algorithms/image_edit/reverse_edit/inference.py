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

    def infer(self, input_path, output_path, prompt, edit_prompt=None):
        image_ori = Image.open(input_path).convert("RGB")
        original_w, original_h = image_ori.size
        target_size = self.pipe.sample_size
        ratio = target_size / max(original_w, original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)
        resized_img = image_ori.resize((new_w, new_h), Image.Resampling.LANCZOS)
        pad_bottom = target_size - new_h
        pad_right = target_size - new_w
        img_np = np.array(resized_img)
        padded_np = np.pad(img_np, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='reflect')
        padded_image = Image.fromarray(padded_np)
        inv_latent, _ = self.pipe.invert(padded_image, prompt, self.config)
        edit_prompt = edit_prompt if edit_prompt is not None else prompt
        rec_image = self.pipe.reconstruct(
            inv_latent,
            edit_prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=1.0,
        )[0]
        rec_image = rec_image.crop((0, 0, new_w, new_h))
        # 恢复到原图大小
        if new_h != target_size or new_w != target_size:
            pass
        if self.use_color_fix and edit_prompt == prompt:
            rec_image = wavelet_color_fix(rec_image, image_ori)   
        rec_image.save(output_path)
        self.pipe.release_all()