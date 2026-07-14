import json
import os
import numpy as np
from PIL import Image
from app.algorithms import general_provider, general_session, general_inference_session, ORTEnvironment
ORTEnvironment.initialize()


IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _resize_bilinear(arr, out_hw):
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((out_hw[1], out_hw[0]), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


class MaskTrackerEdgeTAM:
    def __init__(self, model_dir):
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
        self.tpos_enc = np.load(os.path.join(model_dir, "maskmem_tpos_enc.npy"))
        with open(os.path.join(model_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.num_maskmem = self.meta["num_maskmem"]
        self.max_obj_ptrs = self.meta["max_obj_ptrs_in_encoder"]
        self.mem_dim = self.meta["mem_dim"]
        self.image_size = self.meta["image_size"]

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

    def run(self, mask_path, frames_dir):
        frames_list = sorted([os.path.join(frames_dir, item) for item in os.listdir(frames_dir)])
        assert len(frames_list) >= 1
        bank = {}
        mask_out = os.path.dirname(mask_path)
        image, (H0, W0) = self._load_frame(frames_list[0])
        mask = self._load_mask(mask_path)
        enc = self.image_encoder.run(None, {"image": image})
        pix_feat, hr0, hr1, vfeat, vpos = enc
        obj_ptr = self.mask_encoder.run(None, {"mask": mask, "pix_feat": pix_feat})[0]
        high_res = mask * 20.0 - 10.0
        mm, mmpos = self.mem_encoder.run(None, {"high_res_mask": high_res.astype(np.float32), "pix_feat": pix_feat})
        bank[0] = {"mm": mm, "mmpos": mmpos, "obj_ptr": obj_ptr, "cond": True}
        Image.fromarray((self._mask_to_orig(mask[0, 0], (H0, W0)) * 255).astype(np.uint8)).save(os.path.join(mask_out, f"{os.path.splitext(frames_list[0])[0]}.png"))
        
        for frame_idx in range(1, len(frames_list)):
            image, (H, W) = self._load_frame(frames_list[frame_idx])
            enc = self.image_encoder.run(None, {"image": image})
            pix_feat, hr0, hr1, vfeat, vpos = enc
            sp_mem, sp_pos, tok = self._build_memory(frame_idx, bank)
            image_embed = self.memory_attention.run(None, {
                "vision_feat": vfeat, "vision_pos_embed": vpos,
                "spatial_memory": sp_mem, "spatial_memory_pos": sp_pos,
                "obj_ptr_tokens": tok,
            })[0]

            obj_ptr, high_res = self.image_decoder.run(None, {
                "image_embed": image_embed, "high_res_feat0": hr0, "high_res_feat1": hr1,
            })
            mm, mmpos = self.mem_encoder.run(None, {"high_res_mask": high_res.astype(np.float32), "pix_feat": pix_feat})
            bank[frame_idx] = {"mm": mm, "mmpos": mmpos, "obj_ptr": obj_ptr, "cond": False}
            logits = _resize_bilinear(high_res[0, 0], (H, W))
            tmp_mask = logits > 0
            Image.fromarray((tmp_mask * 255).astype(np.uint8)).save(os.path.join(mask_out, f"{os.path.splitext(frames_list[frame_idx])[0]}.png"))
        return -1

    def _mask_to_orig(self, mask_s, out_hw):
        m = _resize_bilinear(mask_s.astype(np.float32), out_hw)
        return m > 0.5
