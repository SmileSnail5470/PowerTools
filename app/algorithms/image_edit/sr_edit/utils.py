import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import RobertaProcessing


class DiagonalGaussianDistribution:
    def __init__(self, parameters: np.ndarray):
        self.parameters = parameters
        self.mean, self.logvar = np.split(parameters, 2, axis=1)
        self.logvar = np.clip(self.logvar, -30.0, 20.0)
        self.std = np.exp(0.5 * self.logvar)
        self.var = np.exp(self.logvar)

    def sample(self, generator=None) -> np.ndarray:
        if generator is None:
            noise = np.random.randn(*self.mean.shape).astype(self.mean.dtype)
        else:
            noise = generator.standard_normal(self.mean.shape, dtype=np.float32)
            noise = noise.astype(self.mean.dtype, copy=False)
        return self.mean + self.std * noise

    def mode(self) -> np.ndarray:
        return self.mean


class Tokenizer:
    def __init__(self, tokenizer_path: str | Path):
        self.tokenizer_path = Path(tokenizer_path)
        config_path = self.tokenizer_path / "tokenizer_config.json"
        self._config = {}
        if config_path.is_file():
            with open(config_path) as f:
                self._config = json.load(f)
        special_tokens_path = self.tokenizer_path / "special_tokens_map.json"
        self._special_tokens_map = {}
        if special_tokens_path.is_file():
            with open(special_tokens_path) as f:
                self._special_tokens_map = json.load(f)
        tokenizer_json = self.tokenizer_path / "tokenizer.json"
        vocab_json = self.tokenizer_path / "vocab.json"
        merges_txt = self.tokenizer_path / "merges.txt"
        if tokenizer_json.is_file():
            self._tokenizer = HFTokenizer.from_file(str(tokenizer_json))
        elif vocab_json.is_file() and merges_txt.is_file():
            self._tokenizer = HFTokenizer(BPE.from_file(str(vocab_json), str(merges_txt)))
            self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            bos_token = self._resolve_special_token("bos_token", "<|startoftext|>")
            eos_token = self._resolve_special_token("eos_token", "<|endoftext|>")
            bos_id = self._tokenizer.token_to_id(bos_token)
            eos_id = self._tokenizer.token_to_id(eos_token)
            if bos_id is not None and eos_id is not None:
                self._tokenizer.post_processor = RobertaProcessing(
                    sep=(eos_token, eos_id),
                    cls_token=(bos_token, bos_id),
                    add_prefix_space=False,
                    trim_offsets=True,
                )
        else:
            raise FileNotFoundError(f"No tokenizer.json or vocab.json+merges.txt found in {tokenizer_path}")
        self.model_max_length = self._config.get("model_max_length", 77)

    def _resolve_special_token(self, token_key: str, default: str) -> str:
        val = self._special_tokens_map.get(token_key)
        if val is not None:
            if isinstance(val, dict):
                return val.get("content", default)
            return val
        val = self._config.get(token_key)
        if val is not None:
            if isinstance(val, dict):
                return val.get("content", default)
            return val
        return default

    def __call__(self, text: str | list[str], padding: str = "max_length", max_length: int | None = None, truncation: bool = True, return_attention_mask: bool = True) -> dict[str, np.ndarray]:
        if isinstance(text, str):
            text = [text]
        max_len = max_length or self.model_max_length
        pad_token = self._resolve_special_token("pad_token", "<|endoftext|>")
        pad_id = self._tokenizer.token_to_id(pad_token) or 0
        self._tokenizer.enable_padding(
            length=max_len if padding == "max_length" else None,
            pad_id=pad_id,
        )
        if truncation:
            self._tokenizer.enable_truncation(max_length=max_len)
        encodings = self._tokenizer.encode_batch(text)
        input_ids = np.array([enc.ids[:max_len] for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask[:max_len] for enc in encodings], dtype=np.int64)
        result = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
        return result
