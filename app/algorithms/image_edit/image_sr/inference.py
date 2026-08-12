import os
import sys
import math
import random
import time
import gc

import cv2
import numpy as np
from pathlib import Path
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel, OrtValue, preload_dlls
preload_dlls(directory="")
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))

from modules.utils import DiagonalGaussianDistribution, ORTSessionWrapper
from modules.schedulers import Scheduler
from utils import Tokenizer

ONNX_WEIGHTS_NAME = "model.onnx"
_positive = (
    "Cinematic, high-contrast, photo-realistic, 8k, ultra HD, "
    "meticulous detailing, hyper sharpness, perfect without deformations"
)
_negative = (
    "Low quality, blurring, jpeg artifacts, deformed, over-smooth, cartoon, noisy,"
    "painting, drawing, sketch, oil painting"
)


class ImageSpliterNp:
    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1, weight_type="Gaussian"):
        assert weight_type in ["Gaussian", "ones"]
        assert stride <= pch_size
        self.weight_type = weight_type
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf
        self.extra_bs = extra_bs
        bs, chn, height, width = im.shape
        self.true_bs = bs
        self.height_starts_list = self._extract_starts(height)
        self.width_starts_list = self._extract_starts(width)
        self.starts_list = []
        for ii in self.height_starts_list:
            for jj in self.width_starts_list:
                self.starts_list.append([ii, jj])
        self.length = len(self.starts_list)
        self.count_pchs = 0
        self.im_ori = im
        self.im_res = np.zeros([bs, chn, height * sf, width * sf], dtype=np.float32)
        self.pixel_count = np.zeros([bs, chn, height * sf, width * sf], dtype=np.float32)

    def _extract_starts(self, length):
        if length <= self.pch_size:
            return [0]
        starts = list(range(0, length, self.stride))
        for ii in range(len(starts)):
            if starts[ii] + self.pch_size > length:
                starts[ii] = length - self.pch_size
        seen = set()
        unique = []
        for s in starts:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def __len__(self):
        return self.length

    def __iter__(self):
        self.count_pchs = 0
        return self

    def __next__(self):
        if self.count_pchs < self.length:
            index_infos = []
            current_starts_list = self.starts_list[
                self.count_pchs : self.count_pchs + self.extra_bs
            ]
            for ii, (h_start, w_start) in enumerate(current_starts_list):
                w_end = w_start + self.pch_size
                h_end = h_start + self.pch_size
                current_pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]
                if ii == 0:
                    pch = current_pch
                else:
                    pch = np.concatenate([pch, current_pch], axis=0)
                h_start_sr = h_start * self.sf
                h_end_sr = h_end * self.sf
                w_start_sr = w_start * self.sf
                w_end_sr = w_end * self.sf
                index_infos.append([h_start_sr, h_end_sr, w_start_sr, w_end_sr])
            self.count_pchs += len(current_starts_list)
        else:
            raise StopIteration()
        return pch, index_infos

    def update(self, pch_res, index_infos):
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = np.split(pch_res, pch_res.shape[0] // self.true_bs, axis=0)
        assert len(pch_list) == len(index_infos)
        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = pch_list[ii].astype(np.float32)
            weight = self._get_weight(current_pch.shape[-2], current_pch.shape[-1])
            self.im_res[:, :, h_start:h_end, w_start:w_end] += current_pch * weight
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += weight

    @staticmethod
    def _generate_kernel_1d(ksize):
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        if ksize % 2 == 0:
            kernel = cv2.getGaussianKernel(ksize=ksize + 1, sigma=sigma, ktype=cv2.CV_32F)
            kernel = kernel[1:]
        else:
            kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma, ktype=cv2.CV_32F)
        return kernel

    def _get_weight(self, height, width):
        if self.weight_type == "ones":
            return np.ones((1, 1, height, width), dtype=np.float32)
        elif self.weight_type == "Gaussian":
            kernel_h = self._generate_kernel_1d(height).reshape(-1, 1)
            kernel_w = self._generate_kernel_1d(width).reshape(1, -1)
            kernel = kernel_h @ kernel_w
            return kernel.reshape(1, 1, height, width)
        else:
            raise ValueError(f"Unsupported weight type: {self.weight_type}")

    def gather(self):
        assert np.all(self.pixel_count != 0)
        return (self.im_res / self.pixel_count).astype(np.float32)


def retrieve_timesteps_invsr(scheduler, timesteps):
    num_inference_steps = len(timesteps)
    timesteps_arr = np.array(timesteps, dtype=np.float32) - 1
    scheduler.timesteps = timesteps_arr
    sigmas_all = ((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5
    timestep_indices = timesteps_arr.astype(np.int64)
    sigmas = sigmas_all[timestep_indices]
    if hasattr(scheduler.config, "final_sigmas_type"):
        final_sigmas_type = scheduler.config.final_sigmas_type
    else:
        final_sigmas_type = "zero"
    if final_sigmas_type == "sigma_min":
        sigma_last = float(((1 - scheduler.alphas_cumprod[0]) / scheduler.alphas_cumprod[0]) ** 0.5)
    elif final_sigmas_type == "zero":
        sigma_last = 0.0
    else:
        raise ValueError(f"`final_sigmas_type` must be 'zero' or 'sigma_min', got {final_sigmas_type}")
    sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
    scheduler.sigmas = sigmas
    scheduler._step_index = None
    scheduler._begin_index = None
    return scheduler.timesteps, num_inference_steps


def bicubic_resize(image, target_h, target_w):
    n, c, h, w = image.shape
    result = np.zeros((n, c, target_h, target_w), dtype=image.dtype)
    for i in range(n):
        img_hwc = image[i].transpose(1, 2, 0)
        resized = cv2.resize(img_hwc, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        result[i] = resized.transpose(2, 0, 1)
    return result


class ImageSRInference:
    def __init__(self, model_base_dir, provider="CUDAExecutionProvider", noise_predictor_path=None, low_memory=None, num_threads=0):
        self.model_base_dir = Path(model_base_dir)
        self.provider = provider
        self.num_threads = num_threads
        self.configs = self._get_configs()
        self._setup_seed()
        if low_memory is None:
            self.low_memory = (provider == "CPUExecutionProvider")
        else:
            self.low_memory = low_memory
        self._noise_predictor_path = noise_predictor_path
        self._use_gpu = (provider == "CUDAExecutionProvider")
        self._build_model(noise_predictor_path)

    def _get_configs(self):
        return {
            "seed": 12345,
            "basesr": {
                "sf": 4,
                "chopping": {
                    "pch_size": 128,
                    "extra_bs": 1,
                    "weight_type": "Gaussian",
                },
            },
            "bs": 1,
            "timesteps": [200],
            "cfg_scale": 1.0,
            "start_timesteps": 200,
        }

    def _setup_seed(self):
        seed = self.configs["seed"]
        random.seed(seed)
        np.random.seed(seed)

    def _get_session_options(self):
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.low_memory:
            sess_options.enable_cpu_mem_arena = False
            sess_options.enable_mem_pattern = True
        else:
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
        if self.num_threads > 0:
            sess_options.intra_op_num_threads = self.num_threads
            sess_options.inter_op_num_threads = max(1, self.num_threads // 2)
        else:
            cpu_count = os.cpu_count() or 4
            sess_options.intra_op_num_threads = cpu_count
            sess_options.inter_op_num_threads = max(1, cpu_count // 2)
        return sess_options

    def _load_session(self, model_path):
        providers = [self.provider]
        sess_options = self._get_session_options()
        sess_options.add_session_config_entry("session.use_env_allocators", "1")
        provider_option = [
            {
                "arena_extend_strategy": "kSameAsRequested",
            }
        ]
        return ORTSessionWrapper(InferenceSession(str(model_path), providers=providers, sess_options=sess_options, provider_options=provider_option))

    def _ensure_model(self, model_name):
        model = getattr(self, model_name, None)
        if model is not None:
            return model
        if model_name == "noise_predictor":
            path = self._noise_predictor_model_path
        else:
            path = self._model_paths[model_name]
        loaded = self._load_session(path)
        setattr(self, model_name, loaded)
        return loaded

    def _release_model(self, model_name):
        if not self.low_memory:
            return
        if getattr(self, '_keep_all_for_batch', False):
            return
        if model_name == "unet" and getattr(self, '_keep_unet', False):
            return
        model = getattr(self, model_name, None)
        if model is not None:
            setattr(self, model_name, None)
            del model
            gc.collect()

    def _build_model(self, noise_predictor_path=None):
        if self._use_gpu:
            cuda_info = ort.OrtMemoryInfo("Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
            arena_cfg = ort.OrtArenaCfg(0, 1, -1, -1)
            ort.create_and_register_allocator_v2("CUDAExecutionProvider", cuda_info, {}, arena_cfg)

        unet_path = self.model_base_dir / "unet" / ONNX_WEIGHTS_NAME
        vae_encoder_path = self.model_base_dir / "vae_encoder" / ONNX_WEIGHTS_NAME
        vae_decoder_path = self.model_base_dir / "vae_decoder" / ONNX_WEIGHTS_NAME
        text_encoder_path = self.model_base_dir / "text_encoder" / ONNX_WEIGHTS_NAME

        self._model_paths = {
            "unet": unet_path,
            "vae_encoder": vae_encoder_path,
            "vae_decoder": vae_decoder_path,
            "text_encoder": text_encoder_path,
        }

        if self.low_memory:
            self.unet = None
            self.vae_encoder = None
            self.vae_decoder = None
            self.text_encoder = None
            path_str = str(self.model_base_dir)
            if "fp16" in path_str:
                self._dtype = np.float16
            else:
                self._dtype = np.float32
        else:
            self.unet = self._load_session(unet_path)
            self.vae_encoder = self._load_session(vae_encoder_path)
            self.vae_decoder = self._load_session(vae_decoder_path)
            self.text_encoder = self._load_session(text_encoder_path)
            self._dtype = self.unet.dtype

        tokenizer_path = self.model_base_dir / "tokenizer"
        self.tokenizer = Tokenizer(tokenizer_path)

        self.scheduler = Scheduler.from_pretrained(
            str(self.model_base_dir), subfolder="scheduler"
        )

        if noise_predictor_path is None:
            noise_predictor_path = self.model_base_dir / "noise_model.onnx"
        else:
            noise_predictor_path = Path(noise_predictor_path)
        self._noise_predictor_model_path = noise_predictor_path
        if noise_predictor_path.is_file():
            self._has_noise_predictor = True
            if self.low_memory:
                self.noise_predictor = None
            else:
                self.noise_predictor = self._load_session(noise_predictor_path)
        else:
            self.noise_predictor = None
            self._has_noise_predictor = False
            print(
                f"[WARNING] Noise predictor model not found at {noise_predictor_path}. "
                "Using random noise initialization instead. "
                "Run convert_noise_predictor.py to export the noise predictor to ONNX."
            )
        self.vae_scale_factor = 8
        self.vae_scaling_factor = 0.18215

    def _encode_prompt(self, prompt, negative_prompt=None):
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        do_cfg = self.configs["cfg_scale"] > 1.0
        text_encoder = self._ensure_model("text_encoder")
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"]
        text_output = text_encoder(input_ids=input_ids)
        prompt_embeds = text_output.get(
            "last_hidden_state", list(text_output.values())[0]
        )
        if do_cfg:
            uncond_tokens = negative_prompt if negative_prompt else [""] * batch_size
            if isinstance(uncond_tokens, str):
                uncond_tokens = [uncond_tokens] * batch_size
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            uncond_output = text_encoder(input_ids=uncond_input["input_ids"])
            negative_embeds = uncond_output.get(
                "last_hidden_state", list(uncond_output.values())[0]
            )
            prompt_embeds = np.concatenate([negative_embeds, prompt_embeds], axis=0)
        self._release_model("text_encoder")
        return prompt_embeds

    def _encode_image_to_latent(self, image):
        vae_encoder = self._ensure_model("vae_encoder")
        output = vae_encoder(sample=image)
        if "latent_sample" in output:
            latent = output["latent_sample"]
        else:
            params = list(output.values())[0]
            dist = DiagonalGaussianDistribution(params)
            latent = dist.sample()
        self._release_model("vae_encoder")
        return latent * self.vae_scaling_factor

    def _decode_latent_to_image(self, latents):
        latents_scaled = latents / self.vae_scaling_factor
        vae_decoder = self._ensure_model("vae_decoder")
        output = vae_decoder(latent_sample=latents_scaled)
        images = output.get("sample", list(output.values())[0])
        self._release_model("vae_decoder")
        images = (images + 1.0) / 2.0
        return np.clip(images, 0.0, 1.0)

    def _predict_noise(self, image, timestep):
        if not self._has_noise_predictor:
            return None
        noise_predictor = self._ensure_model("noise_predictor")
        sample = image * 2.0 - 1.0
        sample = sample.astype(self._dtype)
        timestep_input = np.array([timestep], dtype=np.float32)
        output = noise_predictor(sample=sample, timestep=timestep_input)
        result = list(output.values())[0]
        self._release_model("noise_predictor")
        if result.shape[1] == 8:
            dist = DiagonalGaussianDistribution(result)
            return dist.sample().astype(np.float32)
        else:
            return result.astype(np.float32)

    def _run_unet(self, latent_input, timestep, prompt_embeds):
        unet = self._ensure_model("unet")
        timestep_input = np.array([timestep], dtype=np.float32)
        output = unet(
            sample=latent_input,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds,
        )
        return output.get("out_sample", list(output.values())[0])

    def _to_gpu(self, arr):
        return OrtValue.ortvalue_from_numpy(arr, "cuda", 0)

    def _encode_image_to_latent_gpu(self, image):
        vae_encoder = self._ensure_model("vae_encoder")
        output = vae_encoder.run_gpu(sample=image)
        if "latent_sample" in output:
            latent_ortval = output["latent_sample"]
        else:
            params = list(output.values())[0].numpy()
            dist = DiagonalGaussianDistribution(params)
            latent = dist.sample()
            latent = (latent * self.vae_scaling_factor).astype(self._dtype)
            self._release_model("vae_encoder")
            return self._to_gpu(latent)
        latent = latent_ortval.numpy()
        latent = (latent * self.vae_scaling_factor).astype(self._dtype)
        self._release_model("vae_encoder")
        return self._to_gpu(latent)

    def _decode_latent_to_image_gpu(self, latents_np):
        latents_scaled = (latents_np / self.vae_scaling_factor).astype(self._dtype)
        vae_decoder = self._ensure_model("vae_decoder")
        output = vae_decoder.run_gpu(latent_sample=latents_scaled)
        images = list(output.values())[0].numpy()
        self._release_model("vae_decoder")
        images = (images + 1.0) / 2.0
        return np.clip(images, 0.0, 1.0)

    def _run_unet_gpu(self, latent_input_ortval, timestep, prompt_embeds_ortval):
        unet = self._ensure_model("unet")
        timestep_input = np.array([timestep], dtype=np.float32)
        output = unet.run_gpu(
            sample=latent_input_ortval,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds_ortval,
        )
        return list(output.values())[0]

    def _invsr_pipeline_gpu(self, im_cond, target_size, prompt_embeds, negative_prompt=None):
        do_cfg = self.configs["cfg_scale"] > 1.0
        target_h, target_w = target_size

        image_up = bicubic_resize(im_cond, target_h, target_w)
        image_up_norm = (image_up * 2.0 - 1.0).astype(self._dtype)

        timesteps, num_inference_steps = retrieve_timesteps_invsr(
            self.scheduler, self.configs["timesteps"]
        )
        latent_timestep = timesteps[0]

        noise = self._predict_noise(im_cond, latent_timestep)

        init_latents_ortval = self._encode_image_to_latent_gpu(image_up_norm)
        init_latents = init_latents_ortval.numpy()

        if noise is None:
            noise = np.random.randn(*init_latents.shape).astype(np.float32)

        sigma = self.scheduler.sigmas[0]
        latents = (init_latents + noise * sigma).astype(self._dtype)

        prompt_embeds_gpu = self._to_gpu(prompt_embeds.astype(self._dtype))

        self._ensure_model("unet")
        for i, t in enumerate(timesteps):
            if self.scheduler._step_index is None:
                self.scheduler._init_step_index(t)

            latent_model_input = (
                np.concatenate([latents, latents], axis=0) if do_cfg else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_input_gpu = self._to_gpu(latent_model_input.astype(self._dtype))
            noise_pred_gpu = self._run_unet_gpu(latent_input_gpu, t, prompt_embeds_gpu)

            noise_pred = noise_pred_gpu.numpy()
            del latent_input_gpu, noise_pred_gpu

            if do_cfg:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + self.configs["cfg_scale"] * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(
                noise_pred.astype(np.float32),
                t,
                latents.astype(np.float32),
                return_dict=False,
            )[0].astype(self._dtype)

        self._release_model("unet")

        result = self._decode_latent_to_image_gpu(latents)
        return result

    def _invsr_pipeline(self, im_cond, target_size, prompt_embeds, negative_prompt=None):
        do_cfg = self.configs["cfg_scale"] > 1.0
        target_h, target_w = target_size
        image_up = bicubic_resize(im_cond, target_h, target_w)
        image_up_norm = image_up * 2.0 - 1.0
        timesteps, num_inference_steps = retrieve_timesteps_invsr(
            self.scheduler, self.configs["timesteps"]
        )
        latent_timestep = timesteps[0]
        noise = self._predict_noise(im_cond, latent_timestep)
        init_latents = self._encode_image_to_latent(image_up_norm.astype(self._dtype))
        if noise is None:
            noise = np.random.randn(*init_latents.shape).astype(np.float32)
        sigma = self.scheduler.sigmas[0]
        latents = init_latents + noise * sigma
        latents = latents.astype(self._dtype)
        self._ensure_model("unet")
        for i, t in enumerate(timesteps):
            if self.scheduler._step_index is None:
                self.scheduler._init_step_index(t)
            latent_model_input = (
                np.concatenate([latents, latents], axis=0)
                if do_cfg
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self._run_unet(
                latent_model_input.astype(self._dtype),
                t,
                prompt_embeds.astype(self._dtype),
            )
            if do_cfg:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + self.configs["cfg_scale"] * (
                    noise_pred_text - noise_pred_uncond
                )
            latents = self.scheduler.step(
                noise_pred.astype(np.float32),
                t,
                latents.astype(np.float32),
                return_dict=False,
            )[0]
            latents = latents.astype(self._dtype)
        self._release_model("unet")
        result = self._decode_latent_to_image(latents.astype(self._dtype))
        return result

    def _run_pipeline(self, im_cond, target_size, prompt_embeds, negative_prompt=None):
        if self._use_gpu:
            return self._invsr_pipeline_gpu(im_cond, target_size, prompt_embeds, negative_prompt)
        return self._invsr_pipeline(im_cond, target_size, prompt_embeds, negative_prompt)

    def sample_func(self, im_cond):
        sf = self.configs["basesr"]["sf"]
        do_cfg = self.configs["cfg_scale"] > 1.0
        if do_cfg:
            negative_prompt = [_negative] * im_cond.shape[0]
        else:
            negative_prompt = None
        ori_h_lq, ori_w_lq = im_cond.shape[-2:]
        ori_w_hq = ori_w_lq * sf
        ori_h_hq = ori_h_lq * sf
        vae_sf = self.vae_scale_factor
        diffusion_sf = 8
        mod_lq = vae_sf // sf * diffusion_sf
        idle_pch_size = self.configs["basesr"]["chopping"]["pch_size"]
        total_pad_h_up = total_pad_w_left = 0
        if min(im_cond.shape[-2:]) < idle_pch_size:
            while min(im_cond.shape[-2:]) < idle_pch_size:
                pad_h_up = max(
                    min((idle_pch_size - im_cond.shape[-2]) // 2, im_cond.shape[-2] - 1), 0
                )
                pad_h_down = max(
                    min(idle_pch_size - im_cond.shape[-2] - pad_h_up, im_cond.shape[-2] - 1), 0
                )
                pad_w_left = max(
                    min((idle_pch_size - im_cond.shape[-1]) // 2, im_cond.shape[-1] - 1), 0
                )
                pad_w_right = max(
                    min(idle_pch_size - im_cond.shape[-1] - pad_w_left, im_cond.shape[-1] - 1), 0
                )
                im_cond = np.pad(
                    im_cond,
                    ((0, 0), (0, 0), (pad_h_up, pad_h_down), (pad_w_left, pad_w_right)),
                    mode="reflect",
                )
                total_pad_h_up += pad_h_up
                total_pad_w_left += pad_w_left

        prompt_embeds = self._encode_prompt([_positive] * im_cond.shape[0], negative_prompt)
        if im_cond.shape[-2] == idle_pch_size and im_cond.shape[-1] == idle_pch_size:
            target_size = (
                im_cond.shape[-2] * sf,
                im_cond.shape[-1] * sf,
            )
            res_sr = self._run_pipeline(
                im_cond,
                target_size=target_size,
                prompt_embeds=prompt_embeds,
                negative_prompt=negative_prompt,
            )
        else:
            if not (im_cond.shape[-2] % mod_lq == 0 and im_cond.shape[-1] % mod_lq == 0):
                target_h_lq = math.ceil(im_cond.shape[-2] / mod_lq) * mod_lq
                target_w_lq = math.ceil(im_cond.shape[-1] / mod_lq) * mod_lq
                pad_h = target_h_lq - im_cond.shape[-2]
                pad_w = target_w_lq - im_cond.shape[-1]
                im_cond = np.pad(
                    im_cond,
                    ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                    mode="reflect",
                )
            im_spliter = ImageSpliterNp(
                im_cond,
                pch_size=idle_pch_size,
                stride=int(idle_pch_size * 0.50),
                sf=sf,
                weight_type=self.configs["basesr"]["chopping"]["weight_type"],
                extra_bs=self.configs["basesr"]["chopping"]["extra_bs"],
            )
            num_patches = len(im_spliter)
            if self.low_memory and num_patches > 1:
                self._ensure_model("unet")
                self._keep_unet = True
                if self._has_noise_predictor:
                    self._ensure_model("noise_predictor")
                self._ensure_model("vae_encoder")
                self._ensure_model("vae_decoder")
                self._keep_all_for_batch = True
            else:
                self._keep_unet = False
                self._keep_all_for_batch = False
            for pch_idx, (im_lq_pch, index_infos) in enumerate(im_spliter):
                start_time = time.time()
                target_size = (
                    im_lq_pch.shape[-2] * sf,
                    im_lq_pch.shape[-1] * sf,
                )
                res_sr_pch = self._run_pipeline(
                    im_lq_pch,
                    target_size=target_size,
                    prompt_embeds=prompt_embeds,
                    negative_prompt=negative_prompt,
                )
                im_spliter.update(res_sr_pch, index_infos)
                print("crop [{}/{}] {} cost: {:.2f}s".format(
                    pch_idx + 1, num_patches, im_lq_pch.shape, time.time() - start_time))
            if self._keep_unet:
                self._keep_unet = False
                self._keep_all_for_batch = False
                self._release_model("unet")
                self._release_model("noise_predictor")
                self._release_model("vae_encoder")
                self._release_model("vae_decoder")
            res_sr = im_spliter.gather()
        total_pad_h_up *= sf
        total_pad_w_left *= sf
        res_sr = res_sr[
            :, :,
            total_pad_h_up : ori_h_hq + total_pad_h_up,
            total_pad_w_left : ori_w_hq + total_pad_w_left,
        ]
        res_sr = np.clip(res_sr, 0.0, 1.0)
        res_sr = res_sr.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        res_sr = (res_sr * 255.0 + 0.5).astype(np.uint8)
        return res_sr

    def inference(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_nchw = img_float.transpose(2, 0, 1)[np.newaxis, ...]
        result = self.sample_func(img_nchw)
        result = result.squeeze(0)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="InvSR ONNX Runtime Inference")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/int8",
        help="Path to ONNX model directory (fp16 for GPU, fp32 for CPU, int8 for quantized CPU)",
    )
    parser.add_argument(
        "--input", type=str,
        default=r"D:\WorkSpace\code\python\modules\blind_watermark_removal\input\test.png",
        help="Input image path",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output image path"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        choices=["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime execution provider",
    )
    parser.add_argument(
        "--noise_predictor",
        type=str,
        default=None,
        help="Path to noise predictor ONNX model",
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        default=None,
        help="Enable low memory mode (auto-enabled for CPU provider). "
             "Models are loaded/unloaded on demand to reduce peak memory.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="Number of CPU threads for inference. 0 = auto (all cores).",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    model_dir = script_dir / args.model_dir

    # Load image
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Cannot read image '{args.input}'")
        sys.exit(1)

    print(f"Provider: {args.provider}")
    print(f"Model dir: {model_dir}")
    print(f"Low memory: {args.low_memory if args.low_memory else 'auto'}")
    print(f"Threads: {args.num_threads if args.num_threads > 0 else 'auto'}")

    total_start = time.time()

    # Create inference object
    sr = ImageSRInference(
        model_base_dir=str(model_dir),
        provider=args.provider,
        noise_predictor_path=args.noise_predictor,
        low_memory=args.low_memory,
        num_threads=args.num_threads,
    )

    # Run inference
    result = sr.inference(img)

    total_time = time.time() - total_start

    # Save output
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_sr{input_path.suffix}"
    else:
        output_path = Path(args.output)

    cv2.imwrite(str(output_path), result)
    print(f"Super-resolved image saved to: {output_path}")
    print(f"Input size: {img.shape[1]}x{img.shape[0]} -> Output size: {result.shape[1]}x{result.shape[0]}")
    print(f"Total time: {total_time:.2f}s")
