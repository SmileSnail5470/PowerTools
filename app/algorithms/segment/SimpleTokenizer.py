"""
A minimal BPE tokenizer matching CLIP's tokenizer, used by SAM3.
"""

import re
from typing import List, Dict, Tuple, Optional


class SimpleTokenizer:
    """BPE tokenizer compatible with CLIP's tokenizer."""

    def __init__(self, vocab_path: str, merges_path: str):
        self.byte_encoder: Dict[int, str] = {}
        self.byte_decoder: Dict[str, int] = {}
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        self.cache: Dict[str, str] = {}

        self._init_byte_encoder()

        # 加载词表 vocab.txt
        with open(vocab_path, "r", encoding="utf-8") as f:
            current_id = 0
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    continue
                self.encoder[line] = current_id
                self.decoder[current_id] = line
                current_id += 1

        # 加载合并规则 merges.txt
        with open(merges_path, "r", encoding="utf-8") as f:
            rank = 0
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    first, second = parts[0], parts[1]
                    self.bpe_ranks[(first, second)] = rank
                    rank += 1

        # SOT / EOT token ids
        self.sot_token_id = self.encoder.get("<|startoftext|>", 49406)
        self.eot_token_id = self.encoder.get("<|endoftext|>", 49407)

    def _init_byte_encoder(self):
        """Initialize byte-to-unicode encoder (same as CLIP)."""
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(0xA1, 0xAC + 1))
        bs += list(range(0xAE, 0xFF + 1))

        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1

        def to_utf8(c: int) -> str:
            if c < 128:
                return chr(c)
            elif c < 2048:
                return chr(0xC0 | (c >> 6)) + chr(0x80 | (c & 0x3F))
            elif c < 65536:
                return (
                    chr(0xE0 | (c >> 12))
                    + chr(0x80 | ((c >> 6) & 0x3F))
                    + chr(0x80 | (c & 0x3F))
                )
            return ""

        self.byte_encoder = {b: to_utf8(c) for b, c in zip(bs, cs)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def bpe(self, token: str) -> str:
        """Apply BPE merging to a single token."""
        if token in self.cache:
            return self.cache[token]

        # Split into individual characters by UTF-8 byte length
        word: List[str] = []
        i = 0
        while i < len(token):
            c = ord(token[i])
            if c >= 0xF0:
                length = 4
            elif c >= 0xE0:
                length = 3
            elif c >= 0xC0:
                length = 2
            else:
                length = 1
            word.append(token[i : i + length])
            i += length

        # Mark end-of-word
        if word:
            word[-1] += "</w>"

        if len(word) == 0:
            return token + "</w>"

        def get_pairs(w: List[str]) -> set:
            pairs = set()
            for i in range(len(w) - 1):
                pairs.add((w[i], w[i + 1]))
            return pairs

        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram: Optional[Tuple[str, str]] = None
            min_rank = -1

            for pair in pairs:
                rank = self.bpe_ranks.get(pair, -1)
                if rank != -1 and (min_rank == -1 or rank < min_rank):
                    min_rank = rank
                    bigram = pair

            if bigram is None or min_rank == -1:
                break

            new_word: List[str] = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == bigram[0]
                    and word[i + 1] == bigram[1]
                ):
                    new_word.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> List[int]:
        """Encode text string into token ids."""
        bpe_tokens: List[int] = []

        cleaned = text.lower()

        # Simplified regex matching the C++ version
        pat = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]+|[^\s a-z0-9]+",
            re.IGNORECASE,
        )

        for match in pat.finditer(cleaned):
            token = match.group()
            # Encode each byte to unicode
            encoded_token = ""
            for b in token.encode("utf-8"):
                encoded_token += self.byte_encoder[b]

            bpe_res = self.bpe(encoded_token)
            for part in bpe_res.split():
                if part in self.encoder:
                    bpe_tokens.append(self.encoder[part])

        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token ids back to text string."""
        text_bytes = b""
        for t in tokens:
            if t in self.decoder:
                token_str = self.decoder[t]
                # Remove </w> markers
                token_str = token_str.replace("</w>", " ")
                # Decode each byte-encoded character
                i = 0
                while i < len(token_str):
                    # Try to find the longest matching byte_decoder key
                    found = None
                    for length in range(1, 5):
                        if i + length <= len(token_str):
                            chunk = token_str[i : i + length]
                            if chunk in self.byte_decoder:
                                found = self.byte_decoder[chunk]
                                i += length
                                break
                    if found is not None:
                        text_bytes += bytes([found])
                    else:
                        text_bytes += token_str[i].encode("utf-8")
                        i += 1

        return text_bytes.decode("utf-8", errors="replace")

    def tokenize(
        self, texts: List[str], context_length: int = 77
    ) -> List[List[int]]:
        """Tokenize a batch of texts, padded to context_length."""
        results: List[List[int]] = []
        for text in texts:
            tokens = self.encode(text)
            result = [0] * context_length
            result[0] = self.sot_token_id
            count = 1
            for t in tokens:
                if count >= context_length - 1:
                    break
                result[count] = t
                count += 1
            result[count] = self.eot_token_id
            results.append(result)
        return results
