from __future__ import annotations

import json
import regex as re

from collections import Counter
from functools import lru_cache


class BPETokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None = None): 
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
            cls, 
            vocab_filepath: str, 
            merges_filepath: str, 
            special_tokens: list[str] | None = None
    ) -> BPETokenizer:
        @lru_cache
        def bytes_to_unicode() -> dict[int, str]:
            bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
            cs = bs[:]
            n = 0
            for b in range(2**8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2**8 + n)
                    n += 1
            characters = [chr(n) for n in cs]
            d = dict(zip(bs, characters))
            return d
        
        def bytes_to_str(b: bytes) -> str:
            byte_to_uni = bytes_to_unicode()
            s = ""
            for bit in b:
                s.join(byte_to_uni[bit])
            return s

        def str_to_bytes(s: str) -> bytes:
            byte_to_uni = bytes_to_unicode()
            byte_decoder = {v: k for k, v in byte_to_uni.items()}
            ans = bytearray()
            for c in s:
                ans.extend([byte_decoder[c]])
            return bytes(ans)

        # 处理vocab
        try:
            with open(vocab_filepath, "r", encoding="utf-8") as f:
                vocab_ = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading vocabulary from {vocab_filepath}: {e}")
        vocab = {v: str_to_bytes(k) for k, v in vocab_.items()}

        # 处理merges
        merges_ = []
        with open(merges_filepath, 'r', encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line and len(clean_line.split(" ")) == 2:
                    merges_.append(tuple(clean_line.split(" ")))
        
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token
        merges = [
            (
                str_to_bytes(str1), 
                str_to_bytes(str2),
            )
            for str1, str2 in merges_
        ]

        return cls(vocab, merges, special_tokens)

    def pre_tokenization(
        self,
        text: str, 
    ) -> list[tuple[bytes]]:
        special_tokens = sorted(self.special_tokens, key=lambda x: -len(x)) if self.special_tokens is not None else []
        escaped_tokens = [re.escape(tok) for tok in special_tokens] if special_tokens else []
        split_pattern = "(" + "|".join(escaped_tokens) + ")"    # 按special_tokens分割input
        match_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        split_texts = re.split(split_pattern, text) if len(escaped_tokens) != 0 else [text]# 得到分割后的文本，格式为list
        pre_tokens = []
        for split_text in split_texts:   
            if self.special_tokens != None and split_text in self.special_tokens:
                pre_tokens.append((split_text.encode('utf-8'),))
            else:
                for word in match_pattern.finditer(split_text):
                    word_str = word.group(0).encode("utf-8")
                    bytes_word_tuple = tuple(bytes([word]) for word in word_str)
                    pre_tokens.append(bytes_word_tuple)
        
        return pre_tokens

    def merge(
            self,
            pre_token: tuple[bytes],
            ranked_merges: dict[bytes, int]
    ) -> tuple[bytes]:
        
        while True:
            cur_min_rank = len(ranked_merges)
            best_pair = None
            for i in range(len(pre_token) - 1):
                pair = pre_token[i] + pre_token[i + 1]
                rk = ranked_merges.get(pair, float('inf'))
                if rk < cur_min_rank:
                    cur_min_rank = rk
                    best_pair = pair
            
            if best_pair is None:
                break
            
            new_token = []
            i = 0
            while i < len(pre_token):
                if i + 1 < len(pre_token) and pre_token[i] + pre_token[i + 1] == best_pair:
                    new_token.append(best_pair)
                    i += 2 
                else:
                    new_token.append(pre_token[i])
                    i += 1
            pre_token = new_token

        return pre_token

    def merge_pre_tokens(
            self,
            pre_tokens: list[tuple[bytes]],
    ) -> list[tuple[bytes]]:
        merged_tokens: list[tuple[bytes]]= []
        special_tokens_bytes = (
            [tuple(special_token.encode('utf-8')) for special_token in self.special_tokens]
            if self.special_tokens else []
        )

        ranked_merges = {bytes1 + bytes2: idx for idx, (bytes1, bytes2) in enumerate(self.merges)}

        for pre_token in pre_tokens:
            if pre_token in special_tokens_bytes:
                merged_tokens.append(pre_token)
            else:
                merged_tokens.append(self.merge(pre_token, ranked_merges))

        return merged_tokens

    def encode(self, text: str) -> list[int]:
        token_to_id = {token: id for id, token in self.vocab.items()}
        tokens = []
        pre_tokens = self.pre_tokenization(text)
        merged_tokens = self.merge_pre_tokens(pre_tokens)
        joined_tokens = []

        for word in merged_tokens:
            for b in word:
                joined_tokens.append(b)
        
        return [token_to_id.get(token, -1) for token in joined_tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            if not chunk:
                continue
            token_ids = self.encode(chunk)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        joined_bytes = bytearray()
        for id in ids:
            joined_bytes.extend(self.vocab[id])
        return bytes(joined_bytes).decode("utf-8", errors="replace")