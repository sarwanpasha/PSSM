"""Minimal native-torch stand-in for the two torch_scatter functions ESM-IF1 imports.

esm.inverse_folding.gvp_modules does `from torch_scatter import scatter_add, scatter`, but the
only call is scatter_add(torch.ones_like(dst), dst, dim_size=n), an occurrence count. torch_scatter
is a compiled extension with no wheels for this platform, so these equivalents use torch primitives
only. Semantics follow torch_scatter: values of `src` are accumulated into `out` at positions given
by `index` along dimension `dim`.
"""
import torch

__all__ = ["scatter_add", "scatter", "scatter_sum", "scatter_mean", "scatter_max"]


def _broadcast(index, src, dim):
    if dim < 0:
        dim = src.dim() + dim
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index = index.view(shape)
    return index.expand_as(src)


def _out_for(src, index, dim, out, dim_size):
    if out is not None:
        return out
    if dim < 0:
        dim = src.dim() + dim
    size = list(src.size())
    size[dim] = int(dim_size) if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
    return torch.zeros(size, dtype=src.dtype, device=src.device)


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    out = _out_for(src, index, dim, out, dim_size)
    d = dim if dim >= 0 else src.dim() + dim
    return out.scatter_add_(d, _broadcast(index, src, dim), src)


scatter_sum = scatter_add


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    total = scatter_add(src, index, dim, None, dim_size)
    count = scatter_add(torch.ones_like(src), index, dim, None, dim_size)
    return total / count.clamp(min=1)


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    d = dim if dim >= 0 else src.dim() + dim
    base = _out_for(src, index, dim, None, dim_size).fill_(float("-inf"))
    res = base.scatter_reduce_(d, _broadcast(index, src, dim), src, reduce="amax", include_self=True)
    return res, None


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    if reduce in ("sum", "add"):
        return scatter_add(src, index, dim, out, dim_size)
    if reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    if reduce in ("max", "amax"):
        return scatter_max(src, index, dim, out, dim_size)[0]
    raise NotImplementedError("reduce=%r is not provided by this stand-in" % reduce)
