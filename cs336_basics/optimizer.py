from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
            
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1

        return loss 


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float =1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float=1e-8, weight_decay: float = 0.1):
        if lr < 0:
            raise ValueError(f"Invaild learning rate {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for w in group['params']:
                if w.grad is None:
                    continue
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                state = self.state[w]
                grad = w.grad.data
                t = state.get('t', 0)
                
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(w.data)
                    state['v'] = torch.zeros_like(w.data)
                
                state['t'] += 1
                t = state['t']
                m, v = state['m'], state['v']
                m.mul_(beta1).add_(grad, alpha=1 - beta1) #v = beta1 * m + (1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) #v = beta2 * m + (1 - beta2) * grad ** 2
                
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
               
                
                alpha = lr * math.sqrt(bias_correction2) / bias_correction1
                w.data.addcdiv_(m, v.sqrt() + eps, value=-alpha)
                
                if weight_decay != 0:
                    w.data.add_(w.data, alpha=-lr * weight_decay)

        return loss
                


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=100)
    
    for t in range(10):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()