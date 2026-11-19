"""Directional ablation (Arditi et al. 2024).

Hooks project a unit direction u out of every write into the residual stream:
the embedding output and every attention (o_proj) and MLP (down_proj) output.
The residual stream is the sum of those writes, so u is absent at every layer
and position. This is mathematically identical to weight orthogonalization
(W <- W - u u^T W for each writer); `orthogonalize_` implements that form for
exporting an ablated model as a normal checkpoint, and tests/test_ablation.py
checks the two agree numerically. With LoRA, the hook sits on the PEFT wrapper,
so it ablates base + adapter output together (equivalently, orthogonalize both
W and lora_B).
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


def writer_modules(decoder) -> list:
    mods = [decoder.embed_tokens]
    for layer in decoder.layers:
        mods += [layer.self_attn.o_proj, layer.mlp.down_proj]
    return mods


@contextmanager
def ablate(decoder, direction: np.ndarray | None):
    """Project `direction` out of the residual stream while the block is active.
    direction=None is a no-op (the unablated baseline goes through the same code)."""
    if direction is None:
        yield
        return
    p = next(decoder.parameters())
    u = torch.tensor(np.asarray(direction, dtype=np.float32), device=p.device)
    u = u / u.norm()

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        proj = (h.float() @ u).unsqueeze(-1) * u
        h2 = (h.float() - proj).to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    # prepend: ablation runs before any other hook on the same module (e.g. activation capture)
    handles = [m.register_forward_hook(hook, prepend=True) for m in writer_modules(decoder)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def orthonormal_basis(directions, tol: float = 1e-6) -> np.ndarray:
    """[d, k] float64 orthonormal columns spanning the rows of `directions` ([k, d] or [d]). Refuses
    linearly dependent rows: QR would then return an arbitrary extra column and project out a direction
    nobody asked for."""
    V = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    Q, R = np.linalg.qr(V.T)
    diag = np.abs(np.diag(R))
    if not np.all(np.isfinite(diag)) or diag.min() <= tol * max(diag.max(), 1e-30):
        raise ValueError(f"directions are linearly dependent or non-finite (|diag R| = {diag.tolist()})")
    return Q


@contextmanager
def ablate_span(decoder, directions):
    """Project span(directions) (rows; orthonormalized here) out of every residual writer, like `ablate`
    for one direction. An empty set is a no-op."""
    V = np.asarray(directions, dtype=np.float64)
    if V.size == 0:
        yield
        return
    p = next(decoder.parameters())
    Q = torch.tensor(orthonormal_basis(V).astype(np.float32), device=p.device)  # [d, k]

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        h2 = (hf - (hf @ Q) @ Q.T).to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    handles = [m.register_forward_hook(hook, prepend=True) for m in writer_modules(decoder)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def orthogonalize_(causal_lm, direction: np.ndarray) -> None:
    """In-place weight orthogonalization of a (possibly PEFT-wrapped) causal LM.
    Unties lm_head from the embedding first if they share storage, so only the
    embedding's *output* is ablated (matching the hook)."""
    base = causal_lm.get_base_model() if hasattr(causal_lm, "get_base_model") else causal_lm
    decoder = base.model
    emb = decoder.embed_tokens.weight
    if base.lm_head.weight.data_ptr() == emb.data_ptr():
        base.lm_head.weight = torch.nn.Parameter(emb.detach().clone())
        base.config.tie_word_embeddings = False
    u = torch.tensor(np.asarray(direction, dtype=np.float32), device=emb.device)
    u = u / u.norm()

    def ortho_out(w):  # w: [d_model, d_in] writes into the residual
        wf = w.float()
        w.copy_((wf - torch.outer(u, u @ wf)).to(w.dtype))

    ef = emb.float()  # [vocab, d_model]
    emb.copy_((ef - torch.outer(ef @ u, u)).to(emb.dtype))
    for layer in decoder.layers:
        for mod in (layer.self_attn.o_proj, layer.mlp.down_proj):
            if hasattr(mod, "base_layer"):  # PEFT LoRA wrapper: orthogonalize base W and every lora_B
                ortho_out(mod.base_layer.weight)
                for lb in mod.lora_B.values():
                    ortho_out(lb.weight)
            else:
                ortho_out(mod.weight)
            bias = getattr(getattr(mod, "base_layer", mod), "bias", None)
            if bias is not None:
                bf = bias.float()
                bias.copy_((bf - (bf @ u) * u).to(bias.dtype))


def random_unit(d: int, seed: int, cov_rows: np.ndarray | None = None) -> np.ndarray:
    """Random unit direction. Isotropic by default; if `cov_rows` (centered activation
    rows, [n, d]) is given, u ~ N(0, cov) via a Gaussian combination of the rows."""
    g = np.random.default_rng(seed)
    if cov_rows is None:
        v = g.standard_normal(d)
    else:
        v = g.standard_normal(cov_rows.shape[0]) @ cov_rows.astype(np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)
