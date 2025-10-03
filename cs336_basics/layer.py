import torch
import math
from torch import nn
from torch.nn.init import trunc_normal_
from einops import einsum, rearrange
from jaxtyping import Float, Int, Bool

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        std = (2 / (in_features + out_features)) ** 0.5
        trunc_normal_(self.W, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, '... in, out in -> ... out')

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        std = 1
        trunc_normal_(self.W, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, token_ids: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len d_model"]:
        return self.W[token_ids]

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        normalized_x = x / RMS
        results = normalized_x * self.W
        return results.to(in_dtype)

def SiLU(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.linear3 = Linear(d_model, d_ff)

    
        
    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        xW1 = self.linear1(x)
        xW3 = self.linear3(x)
        return self.linear2(SiLU(xW1) * xW3)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device = None
    ):
        super().__init__()
        self.theta = theta
        assert d_k % 2 == 0, 'd_k is not even'
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        factory_kwargs = {'device': device}

        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(0, max_seq_len, device=device).float()
        freqs = torch.outer(positions, freq)
        self.register_buffer('cos_cache', torch.cos(freqs.clone().detach()))        
        self.register_buffer('sin_cache', torch.sin(freqs.clone().detach()))   

    def forward(
            self,
            x: Float[torch.Tensor, "... seq_len d_k"],
            token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        
        return out

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

class ScaledDotProductAttention(nn.Module):
    def __init__(
            self,
            d_k: int
    ):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(
            self,
            q: Float[torch.Tensor, "... seq_len_q d_k"],
            k: Float[torch.Tensor, "... seq_len_k d_k"],
            v: Float[torch.Tensor, "... seq_len_k d_v"],
            mask: Bool[torch.Tensor, "seq_len_q seq_len_k"] | None = None
    ) -> Float[torch.Tensor, "... seq_len_q d_v"]:
        attention_score = einsum(q, k, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") * self.scale
        if mask is not None:
            attention_score = attention_score.masked_fill(~mask, float("-inf"))
        attention_score = softmax(attention_score, dim=-1)
        return einsum(attention_score, v, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")

class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int | None = None,
            theta: float = 10000.0,
            use_rope: bool = True,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model can not divide num_heads"
        factory_kwargs = {'device': device, 'dtype': dtype}
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.mask = mask.unsqueeze(0).unsqueeze(0)
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.use_rope = use_rope
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.attn = ScaledDotProductAttention(self.d_k)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [Linear(d_model, d_model, **factory_kwargs) for i in range(4)]      
        if use_rope is True:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device
            )
        

    def forward(
            self, 
            x: Float[torch.Tensor, "batch_size seq_len d_model"],
            token_positions: Int[torch.Tensor, " batch_size sequence_length"] | None = None,
    ) -> Float[torch.Tensor, "batch_size seq_len d_model"]:
        B, S, _ = x.shape 
        q, k, v = [rearrange(proj(x), 'b s (h d) -> b h s d', h = self.num_heads) for proj in [self.q_proj, self.k_proj, self.v_proj]]
        if self.use_rope is True:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        out = self.attn(q, k, v, self.mask[..., :S, :S])
        return self.o_proj(rearrange(out, 'b h s d -> b s (h d)'))

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float = 10000.0,
            use_rope: bool = True,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.attn = CausalMultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=use_rope,
            device=device,
            dtype=dtype
        )
        self.norm2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        
    def forward(
                self, 
                x: Float[torch.Tensor, " batch seq_len d_model"],
                token_positions: Int[torch.Tensor, "batch seq_len"]
    ) -> Float[torch.Tensor, " batch seq_len d_model"]:
        b, s, _ = x.shape
        attn_out = self.attn(self.norm1(x), token_positions=token_positions)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x

def _copy_param(target: torch.Tensor, source: torch.Tensor) -> None:
    """
    Copy `source` into `target` in-place, transposing `source` if that
    is what makes the shapes line up.
    """
    if source.shape == target.shape:
        target.data.copy_(source)
    elif source.T.shape == target.shape:
        target.data.copy_(source.T)
    else:
        raise ValueError(f"Shape mismatch: cannot load parameter of shape {source.shape} "
                         f"into tensor of shape {target.shape}")

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            theta: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        factory_kwargs = {"device": device, "dtype": dtype}
        self.emb = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            **factory_kwargs
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                use_rope=True,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.final_linear = Linear(in_features=d_model, out_features=vocab_size, **factory_kwargs)


    def forward(self, token_ids: Int[torch.Tensor, "batch seq_len"]) -> Float[torch.Tensor, "batch seq_len vocab_size"]:
        B, S = token_ids.shape
        assert S < self.context_length, "text is too long"
        x = self.emb(token_ids)
        pos = torch.arange(S).unsqueeze(0).expand(B, S)
        for block in self.blocks:
            x = block(x, pos)
        x = self.final_norm(x)
        x = self.final_linear(x)
        # logits = softmax(x, dim=-1)
        return x

