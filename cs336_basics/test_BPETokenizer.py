import pytest
import sys
import tiktoken
import train_bpe

from pathlib import Path
from BPETokenizer import BPETokenizer

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))


@pytest.fixture
def tokenizer():    
    return BPETokenizer.from_files(str(base_dir / 'vocab.json'), str(base_dir / 'merges.txt'), ['<|endoftext|>'])
    
def test_from_files(tokenizer):
    a = isinstance(tokenizer.vocab, dict) and all(isinstance(k, int) and isinstance(v, bytes) for k, v in tokenizer.vocab.items())
    b = isinstance(tokenizer.merges, list) and all(isinstance(item, tuple) for item in tokenizer.merges)
    c = isinstance(tokenizer.special_tokens, list) and all(isinstance(item, str) for item in tokenizer.special_tokens)
    assert a and b and c, "vocab type error"

def test_pre_tokenization(tokenizer):
    ## pre_tokenization step
    with open(str(base_dir.parent/ 'tests' / 'fixtures' / 'tinystories_sample.txt'), 'r', encoding='utf-8') as f:
        input_text = f.read()
    # print(tokenizer.pre_tokenization(input_text))
    # assert tokenizer.pre_tokenization(input_text) == train_bpe.pre_tokenization(input_text, ['endoftext']), "pre_tokenization error"
    # assert 1

def test_encode(tokenizer):
    with open(str(base_dir.parent/ 'tests' / 'fixtures' / 'tinystories_sample.txt'), 'r', encoding='utf-8') as f:
        input_text = f.read()
    a = tokenizer.encode(input_text)
    # print(sum([1 for item in a if item == -1]))
    # print(a)

def test_decode(tokenizer):
    print(tokenizer.decode([10, 256, 10]))

def test_encode_decode_empty(tokenizer):
    token_ids =  tokenizer.encode("")
    print("" == tokenizer.decode(token_ids))


    