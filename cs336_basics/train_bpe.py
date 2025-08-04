import os
import regex as re
from pprint import pprint
from tqdm import tqdm
from typing import BinaryIO
from multiprocessing import Pool
from collections import Counter
import json
from pathlib import Path

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## initialize vocabulary step
def initialize_vocabulary(
        special_tokens: list[str]
) -> dict[int, bytes]:
    vocabulary = {}
    vocabulary.update({i: special_tokens[i].encode("utf-8") for i in range(0, len(special_tokens))})
    vocabulary.update({i + len(vocabulary): bytes([i]) for i in range(256)})  
    
    return vocabulary


## pre_tokenization step
def pre_tokenization(
        input: str, 
        special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    escaped_tokens = [re.escape(tok) for tok in special_tokens]
    split_pattern = "|".join(escaped_tokens) # 按special_tokens分割input
    match_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""") # 分割后匹配除去special_tokens中的word

    split_texts = re.split(split_pattern, input) # 得到分割后的文本，格式为list
    pre_tokens = {}
    for split_text in split_texts:
        for word in match_pattern.finditer(split_text):
            word_str = word.group(0).encode("utf-8")
            bytes_word_tuple = tuple(bytes([word]) for word in word_str)
            pre_tokens[bytes_word_tuple] = pre_tokens.get(bytes_word_tuple, 0) + 1 
    
    return pre_tokens


def merge_pre_tokens(
        dicts: list[Counter[tuple[bytes]]]
) -> Counter[tuple[bytes]]:
    merged_counter = Counter()
    for counter in dicts:
        merged_counter.update(counter)
    return merged_counter



## 多进程进行pre_tokenization
def parallel_pre_tokenization(
        file_path: str, 
        special_tokens: list[str], 
        num_workers: int = None
) -> Counter[tuple[bytes]]:
    params = []
    with open(file_path, 'rb') as f:
        boundary = find_chunk_boundaries(f, num_workers, special_tokens[0].encode("utf-8")) 
        for left, right in zip(boundary[:-1], boundary[1:]):
            f.seek(left)
            chunk = f.read(right - left).decode("utf-8", errors="ignore")
            params.append((chunk, special_tokens))
    with Pool(processes=num_workers) as pool:
        result_dicts = pool.starmap(pre_tokenization, params)

    return merge_pre_tokens(result_dicts)


## merge_tools
def get_merged_word(
        word: tuple[bytes], 
        cmp_pair: tuple[bytes]
) -> tuple[bytes]:
    new_word = [] # 存储merge后的word
    length, cur = len(word), 0
    while cur < length:
        if cur + 1 < length: # 当还能组成的pair时
            if (word[cur], word[cur + 1]) == cmp_pair: # 找到了可以merge的对象
                new_word.append(word[cur] + word[cur + 1])
                cur += 2
            else:
                new_word.append(word[cur])
                cur += 1    
        else:
            new_word.append(word[cur])
            cur += 1
    return tuple(new_word)


def get_pair_freq(
        word_counts: Counter[tuple[bytes]]
) -> Counter[tuple[bytes]]:
    freq_pair: Counter[tuple[bytes]] = {}
    for word, cnt in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            freq_pair[pair] = freq_pair.get(pair, 0) + cnt
    return freq_pair

## merge_tools 
def find_pair(
        freq_pair: Counter[tuple[bytes]]
) -> tuple[bytes]:
    
    max_value = max(freq_pair.values())
    max_pair = max([k for k, v in freq_pair.items() if v == max_value])
    return max_pair


def train_bpe(
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    ## setp1 initinalize vocabulary
    vocabulary: dict[int, bytes] = initialize_vocabulary(special_tokens)

    ## setp2 pre tokenization
    # file_path = "assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    word_counts = parallel_pre_tokenization(
            input_path,
            special_tokens,
            16
    )

    cur_id: int = len(vocabulary)
    merges: list[tuple[bytes, bytes]] = []
    ## step3 BPE merge
    need_merge_cnt: int = vocab_size - cur_id

    pair_freqs  = get_pair_freq(word_counts)

    for i in tqdm(range(need_merge_cnt)): # 迭代merge频次最高的byte-pair

        if not pair_freqs:
            break
        best_pair = find_pair(pair_freqs)
        merges.append(best_pair)
        vocabulary[cur_id] = best_pair[0] + best_pair[1]
        cur_id += 1

        # 找出所有需要更新的word
        words_need_update = {}
        for word, cnt in word_counts.items():
            if best_pair[0] in word and best_pair[1] in word:
                for i in range(len(word) - 1):
                    if (word[i], word[i + 1]) == best_pair:
                        words_need_update[word] = cnt
                        break

        # 更新word_counts
        for word, cnt in words_need_update.items():
            # 增量更新pair频率表
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) - cnt

            del word_counts[word]
            new_word = get_merged_word(word, best_pair)
            word_counts[new_word] = word_counts.get(new_word, 0) + cnt

            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + cnt
    
    return vocabulary, merges

from functools import lru_cache
@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
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
        s += byte_to_uni[bit] 
    return s

def str_to_bytes(s: str) -> bytes:
    byte_to_uni = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_to_uni.items()}
    ans = bytes()
    for c in s:
        ans += bytes([byte_decoder[c]])
    return ans

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    data_path = Path(__file__).resolve().parent.parent/ "data" / "TinyStoriesV2-GPT4-train.txt"
    vocabulary, merges = train_bpe(str(data_path), 10000, ["<|endoftext|>"])
    pprint(vocabulary)
    byte_to_uni = bytes_to_unicode()
    
    byte_to_char = {}
    byte_decoder = {v: k for k, v in byte_to_uni.items()}
    vocab = {bytes_to_str(v): k for k, v in vocabulary.items()}
    pprint(vocab)
    
    vocab_result_path = base_dir / "vocab.json"
    merges_result_path = base_dir / "merges.txt"
    vocab_result_path.touch()
    merges_result_path.touch()
    with open(str(vocab_result_path), "w") as f:
        json.dump(vocab, f, indent=4)
    with open(str(merges_result_path), "w", encoding="utf-8") as f:
        for byte1, byte2 in merges:
            f.write(bytes_to_str(byte1) + " " + bytes_to_str(byte2) + '\n')


