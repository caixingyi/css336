import torch
import math

from torch import nn
from torch import Tensor
from einops import einsum, rearrange
from jaxtyping import Float, Int


def cross_entropy(o_i: Float[Tensor, "... vocab_size"], x: Tensor) -> Tensor:
    ## 和softmax一样为了数值稳定
    assert o_i.dim() in (2, 3), "dim error"
    if o_i.dim() == 3:
        o_i = rearrange(o_i, 'B S V -> (B S) V')
        x = rearrange(x, 'B S -> (B S)')
    B, V = o_i.shape
    o_max = torch.max(o_i, dim=-1, keepdim=True).values
    
    shift_o = o_i - o_max
    exp_o = torch.exp(shift_o)
    target_logit = shift_o[torch.arange(B), x]
    sum_logit = torch.sum(exp_o, dim=-1)

    # 这里有一些数学运算上面的技巧
    loss = torch.log(sum_logit) - target_logit
    
    return loss.mean()

def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c) -> float:
    alpha = 0
    if t < T_w:
        alpha = t * alpha_max / T_w
    elif t >= T_w and t <= T_c:
        alpha = alpha_min + 0.5 * (1 + math.cos((t - T_w) * math.pi / (T_c - T_w))) * (alpha_max - alpha_min)
    else:
        alpha = alpha_min

    return alpha

def gradient_clipping(params: list[Tensor], M: float, eps: float = 1e-6) -> None:
    # 原地操作不需要返回
    flattened_grads = [p.grad.flatten() for p in params if p is not None and p.grad is not None]
    all_grads = torch.cat(flattened_grads, dim=0)
    norm = torch.norm(all_grads)
    if norm >= M:
        for p in params:
            if p is not None and p.grad is not None:
                p.grad *= M/(norm + eps)
