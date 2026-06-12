"""Muon optimizer (single-GPU) with Moonlight-style RMS matching.

Muon orthogonalizes the momentum of 2D hidden-layer weight matrices via
Newton-Schulz iteration (Jordan et al. 2024). The `0.2 * sqrt(max(m, n))`
update scaling (Liu et al. 2025, "Muon is Scalable for LLM Training") matches
AdamW's typical update RMS, so the same lr / schedule / weight decay can be
shared with the AdamW group handling embeddings, head, norms and biases.

Usage:
    muon_params, adamw_params = split_params(model)
    opt = torch.optim.AdamW(adamw_params, lr=3e-4, weight_decay=0.1)
    muon = Muon(muon_params, lr=3e-4, weight_decay=0.1)
    # call muon.step() alongside opt.step(); share one LR scheduler via
    # scheduler-driven lr assignment on both (both expose param_groups).
"""

import math

import torch


@torch.no_grad()
def newton_schulz5(g, steps=5, eps=1e-7):
    """Approximate UV^T (orthogonalization) of a 2D matrix via 5 NS iterations."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(torch.bfloat16)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden-layer weights. Do NOT put embeddings/head/norms here."""

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                assert p.ndim == 2, f"Muon needs 2D params, got {p.shape}"

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p.grad)
                buf = state["buf"]
                buf.mul_(mu).add_(p.grad)
                g = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf
                u = newton_schulz5(g, group["ns_steps"])
                # match AdamW update RMS so lr can be shared (Moonlight)
                scale = 0.2 * math.sqrt(max(p.shape))
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"] * scale)
        return loss


def split_params(model):
    """(muon_params, adamw_params): 2D hidden weights -> Muon, rest -> AdamW.

    Embedding tables, the LM head, norms, biases and 1D params stay on AdamW.
    """
    emb_like = set()
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding):
            emb_like.add(id(mod.weight))
    head = getattr(model, "head", None)
    if head is not None:
        emb_like.add(id(head.weight))
    muon_params, adamw_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and id(p) not in emb_like:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
