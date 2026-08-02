from PIL import Image
from app.algorithms.image_edit.color_fix import wavelet_color_fix
from app.algorithms.image_edit.reverse_edit.pipeline import ORTPipeline, RunConfig


class ReverseEditInference:
    def __init__(
        self,
        model_dir,
        low_memory: bool = True,
        use_io_binding: bool | None = None,
        use_cupy: bool = True,
        intra_op_num_threads: int | None = None,
        num_inference_steps: int = 50,
        num_inversion_steps: int = 50,
        num_renoise_steps: int = 1,
        guidance_scale: float = 0.0,
        lambda_ac: float = 20.0,
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
        self.num_inference_steps = num_inference_steps
        self.num_inversion_steps = num_inversion_steps
        self.num_renoise_steps = num_renoise_steps
        self.guidance_scale = guidance_scale
        self.lambda_ac = lambda_ac
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
        self.config = RunConfig(
                num_inference_steps=self.num_inference_steps,
                num_inversion_steps=self.num_inversion_steps,
                num_renoise_steps=self.num_renoise_steps,
                guidance_scale=self.guidance_scale,
                noise_regularization_lambda_ac=self.lambda_ac,
                perform_noise_correction=False,
            )

    def infer(self, input_path, output_path, prompt, edit_prompt=None):
        image_ori = Image.open(input_path).convert("RGB")
        image = image_ori.resize((self.pipe.sample_size, self.pipe.sample_size))
        inv_latent, _ = self.pipe.invert(image, prompt, self.config)
        if not self.no_reconstruction:
            edit_prompt = edit_prompt if edit_prompt is not None else prompt
            rec_image = self.pipe.reconstruct(
                inv_latent,
                edit_prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=1.0,
            )[0]
            rec_image = rec_image.resize(image_ori.size, Image.Resampling.LANCZOS)
            rec_image = wavelet_color_fix(rec_image, image_ori) if self.use_color_fix else rec_image
            rec_image.save(output_path)
        self.pipe.release_all()