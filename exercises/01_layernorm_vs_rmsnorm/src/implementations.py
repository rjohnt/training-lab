"""Bias-free normalization families; each norm has its own FP64 reference."""
import torch
import torch.nn.functional as F


def formula(x, weight, eps, norm, high_precision=False):
    z = x.double() if high_precision else x.float()
    w = weight.double() if high_precision else weight.float()
    if norm == "layernorm":
        z = z - z.mean(dim=-1, keepdim=True)
    y = z * torch.rsqrt((z * z).mean(dim=-1, keepdim=True) + eps) * w
    return y if high_precision else y.to(x.dtype)


def build(norm, family, eps):
    if family == "native":
        if norm == "layernorm":
            return lambda x, w: F.layer_norm(x, (x.shape[-1],), w, None, eps)
        return lambda x, w: F.rms_norm(x, (x.shape[-1],), w, eps)
    def fn(x, w):
        return formula(x, w, eps, norm)
    if family == "eager":
        return fn
    if family == "compiled":
        return torch.compile(fn, fullgraph=True, dynamic=False,
                             options={"triton.cudagraphs": False})
    raise ValueError(family)
