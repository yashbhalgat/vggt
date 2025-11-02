import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16


@dataclass
class AttnContext:
    B: int
    S: int
    P_frame: int
    patch_start_idx: int
    H_p: int
    W_p: int


def load_model() -> VGGT:
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model = model.to(device)
    model.eval()
    return model


def run_oracle(model: VGGT, images: torch.Tensor) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        if dtype is not None and images.is_cuda:
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = model(images)
        else:
            preds = model(images)

    out: Dict[str, np.ndarray] = {}
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
    return out


def _mass_preserving_mask(values: torch.Tensor, mass_fraction: float) -> torch.Tensor:
    """
    values: 1D tensor of non-negative attention masses per key (length N)
    returns boolean mask keep_mask (True = keep), shape [N]
    """
    v = values.float().clamp(min=0)
    total = v.sum()
    if total <= 0:
        return torch.zeros_like(v, dtype=torch.bool)
    frac = float(np.clip(mass_fraction, 0.0, 1.0))
    if frac <= 0.0:
        return torch.zeros_like(v, dtype=torch.bool)
    target = frac * total
    # sort descending
    val, idx = torch.sort(v, descending=True)
    csum = torch.cumsum(val, dim=0)
    cutoff_idx = int(torch.searchsorted(csum, torch.as_tensor(target, device=csum.device), right=False).item())
    if cutoff_idx >= v.numel():
        keep = torch.ones_like(v, dtype=torch.bool)
    else:
        thr = val[cutoff_idx]
        keep = v >= thr
    return keep


def _get_attn_context(model: VGGT) -> AttnContext:
    agg = model.aggregator
    ctx = getattr(agg, "_attn_context", None)
    if ctx is None:
        raise RuntimeError("Missing _attn_context; ensure a forward pass occurred to set it")
    H_p, W_p = ctx.get("HW_patches", (0, 0))
    return AttnContext(
        B=int(ctx.get("B", 0)),
        S=int(ctx.get("S", 0)),
        P_frame=int(ctx.get("P_frame", 0)),
        patch_start_idx=int(ctx.get("patch_start_idx", 0)),
        H_p=int(H_p),
        W_p=int(W_p),
    )


def compute_reference_key_keep_mask(model: VGGT, ref_layer: int, mass_fraction: float) -> torch.Tensor:
    """
    Returns a boolean keep mask over keys (length N = S*P_frame), where True means keep key.
    The mask is derived from head-averaged attention from camera-token queries to all keys
    in the reference global layer, aggregated over camera queries.
    """
    agg = model.aggregator
    blocks = getattr(agg, "global_blocks", [])
    if ref_layer < 0 or ref_layer >= len(blocks):
        raise ValueError(f"ref_layer {ref_layer} out of range (0..{len(blocks)-1})")

    keep_mask_out: Dict[str, torch.Tensor] = {}

    def attn_cb(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, module) -> None:
        # q,k,v: [B, H, N, D]
        ctx = _get_attn_context(model)
        B = ctx.B
        S = ctx.S
        P_frame = ctx.P_frame
        patch_start_idx = ctx.patch_start_idx
        if B <= 0 or S <= 0 or P_frame <= 0:
            return
        device = q.device
        head_dim = q.shape[-1]
        # camera token indices (first token per frame)
        cam_idx = torch.arange(S, device=device) * P_frame
        q_cam = q[:, :, cam_idx, :].to(torch.float32)  # [B,H,S,D]
        k_all = k.to(torch.float32)  # [B,H,N,D]
        logits = torch.einsum("bhsd,bhnd->bhsn", q_cam, k_all) / np.sqrt(head_dim)
        attn = torch.softmax(logits, dim=-1)  # [B,H,S,N]
        # average over heads and camera queries, then over batch -> per-key importance
        per_key = attn.mean(dim=(0, 1, 2))  # [N]
        keep_mask = _mass_preserving_mask(per_key, mass_fraction)
        keep_mask_out["keep"] = keep_mask.detach().to("cpu")

    # Install callback on the reference layer and run a forward to collect q/k
    target_attn = blocks[ref_layer].attn
    prev_cb = getattr(target_attn, "attn_callback", None)
    target_attn.attn_callback = attn_cb
    try:
        # A lightweight forward with no changes; images must already be in memory
        # We rely on caller to supply images via outer context
        pass
    finally:
        # Restore even if collection fails
        target_attn.attn_callback = prev_cb

    # The callback will be triggered during model forward. Return after external forward.
    # Caller must check keep_mask_out.
    return keep_mask_out.get("keep")


class MaskedAttentionWrapper:
    def __init__(self, model: VGGT, key_keep_mask_cpu: np.ndarray, ref_layer: int):
        self.model = model
        self.keep_mask = torch.from_numpy(key_keep_mask_cpu.astype(np.bool_))
        self.ref_layer = ref_layer
        self._orig_forwards: List[Optional[Callable]] = []

    def _make_forward(self, attn_module) -> Callable:
        keep_mask_cpu = self.keep_mask  # [N] bool

        def forward(x: torch.Tensor, pos=None) -> torch.Tensor:
            B, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attn_module.q_norm(q), attn_module.k_norm(k)
            if attn_module.rope is not None:
                q = attn_module.rope(q, pos)
                k = attn_module.rope(k, pos)

            # Build additive key mask of shape [1,1,1,N] (0 keep, -inf drop) to avoid NxN logits
            ctx = _get_attn_context(self.model)
            S, P_frame, patch_start_idx = ctx.S, ctx.P_frame, ctx.patch_start_idx
            km = keep_mask_cpu.clone()
            if km.numel() != N:
                km = torch.ones(N, dtype=torch.bool)
            # Always keep special tokens (camera/register) before patches in each frame
            for s in range(S):
                base = s * P_frame
                if base < N:
                    end = min(base + patch_start_idx, N)
                    km[base:end] = True
            add_col = torch.where(
                km.to(x.device),
                torch.zeros(N, device=x.device, dtype=q.dtype),
                torch.full((N,), -1e9, device=x.device, dtype=q.dtype),
            ).view(1, 1, 1, N)

            if getattr(attn_module, "fused_attn", True):
                x_attn = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=add_col,
                    dropout_p=attn_module.attn_drop.p if attn_module.training else 0.0,
                )
            else:
                scale = float(attn_module.head_dim) ** -0.5
                q_ = q.to(torch.float32) * scale
                k_ = k.to(torch.float32)
                logits = torch.matmul(q_, k_.transpose(-2, -1))  # [B,H,N,N]
                logits = logits + add_col.squeeze(0).squeeze(0)  # broadcast over K
                attn = torch.softmax(logits, dim=-1).to(v.dtype)
                x_attn = torch.matmul(attn, v)

            x_out = x_attn.transpose(1, 2).reshape(B, N, C)
            x_out = attn_module.proj(x_out)
            x_out = attn_module.proj_drop(x_out)
            return x_out

        return forward

    @contextmanager
    def install(self):
        blocks = getattr(self.model.aggregator, "global_blocks", [])
        self._orig_forwards = []
        try:
            for i, blk in enumerate(blocks):
                if i <= self.ref_layer:
                    self._orig_forwards.append(None)
                    continue
                attn = getattr(blk, "attn", None)
                if attn is None:
                    self._orig_forwards.append(None)
                    continue
                self._orig_forwards.append(attn.forward)
                attn.forward = self._make_forward(attn)
            yield
        finally:
            # restore
            for i, blk in enumerate(blocks):
                attn = getattr(blk, "attn", None)
                if attn is None:
                    continue
                orig = self._orig_forwards[i] if i < len(self._orig_forwards) else None
                if orig is not None:
                    attn.forward = orig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--image_set_name", type=str, default="kitchen")
    parser.add_argument("--image_extension", type=str, default="png")
    parser.add_argument("--ref_layers", type=str, default="12", help="Comma-separated ref layers, e.g., 12,13,14")
    parser.add_argument("--masses", type=str, default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--output_dir", type=str, default="perf_degradation_out")
    args = parser.parse_args()

    model = load_model()

    # Load images
    if args.image_dir is not None:
        import os
        from pathlib import Path as _P
        image_dir = _P(args.image_dir)
    else:
        from pathlib import Path as _P
        image_dir = _P(f"examples/{args.image_set_name}/images")
    image_paths = sorted([str(p) for p in image_dir.glob(f"*.{args.image_extension}")])
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    images = load_and_preprocess_images(image_paths).to(device)

    # Pass 1: Oracle (saved once)
    oracle = run_oracle(model, images)

    # Oracle saved later; per-reference captures happen below in the loop

    # Prepare outputs directory
    import os
    from pathlib import Path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save oracle once (ref-layer independent)
    np.savez(out_dir / "oracle_outputs.npz", **oracle)

    # Parse masses
    # Parse masses and reference layers
    try:
        mass_list = [float(x.strip()) for x in args.masses.split(",") if x.strip()]
    except Exception:
        mass_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    try:
        ref_layers = [int(x.strip()) for x in args.ref_layers.split(",") if x.strip()]
    except Exception:
        ref_layers = [12]

    # Aggregate summaries
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    plot_data: Dict[int, Tuple[List[float], List[float]]] = {}

    # For each reference layer, capture per-key importances once, then sweep masses
    for ref_layer in ref_layers:
        # Capture per-key vector at this ref layer
        per_key_cpu: Optional[torch.Tensor] = None

        def ref_attn_cb(q, k, v, module):
            nonlocal per_key_cpu
            if per_key_cpu is not None:
                return
            ctx = _get_attn_context(model)
            B, S, P_frame = ctx.B, ctx.S, ctx.P_frame
            if B <= 0 or S <= 0 or P_frame <= 0:
                return
            device_local = q.device
            head_dim = q.shape[-1]
            cam_idx = torch.arange(S, device=device_local) * P_frame
            q_cam = q[:, :, cam_idx, :].to(torch.float32)
            k_all = k.to(torch.float32)
            logits = torch.einsum("bhsd,bhnd->bhsn", q_cam, k_all) / np.sqrt(head_dim)
            attn = torch.softmax(logits, dim=-1)
            per_key_cpu = attn.mean(dim=(0, 1, 2)).detach().cpu()

        blocks = getattr(model.aggregator, "global_blocks", [])
        if ref_layer < 0 or ref_layer >= len(blocks):
            raise ValueError(f"ref_layer {ref_layer} out of range (0..{len(blocks)-1})")
        ref_attn = blocks[ref_layer].attn
        prev_cb = getattr(ref_attn, "attn_callback", None)
        ref_attn.attn_callback = ref_attn_cb
        with torch.no_grad():
            if dtype is not None and images.is_cuda:
                with torch.cuda.amp.autocast(dtype=dtype):
                    _ = model(images)
            else:
                _ = model(images)
        ref_attn.attn_callback = prev_cb
        if per_key_cpu is None:
            raise RuntimeError(f"Failed to compute per-key importances at reference layer {ref_layer}")

        metrics_by_mass: Dict[str, Dict[str, float]] = {}
        masses_for_plot: List[float] = []
        mae_points: List[float] = []

        for m in mass_list:
            m_clamped = float(np.clip(m, 0.0, 1.0))
            keep_mask = _mass_preserving_mask(per_key_cpu.to(torch.float32), m_clamped)
            wrapper = MaskedAttentionWrapper(model, keep_mask.numpy(), ref_layer)
            with wrapper.install():
                masked = run_oracle(model, images)

            mass_tag = ("{:.2f}".format(m_clamped)).replace(".", "p")
            np.savez(out_dir / f"masked_outputs_refL{ref_layer}_mass{mass_tag}.npz", **masked)

            summary: Dict[str, float] = {}
            if "depth" in oracle and "depth" in masked:
                d1 = oracle["depth"].astype(np.float32)
                d2 = masked["depth"].astype(np.float32)
                summary["depth_mae"] = float(np.mean(np.abs(d1 - d2)))
            if "world_points" in oracle and "world_points" in masked:
                p1 = oracle["world_points"].astype(np.float32)
                p2 = masked["world_points"].astype(np.float32)
                wp_mae = float(np.mean(np.abs(p1 - p2)))
                summary["world_points_mae"] = wp_mae
                mae_points.append(wp_mae)
                masses_for_plot.append(m_clamped)
            metrics_by_mass[str(m_clamped)] = summary

        # Save per-ref summary
        with open(out_dir / f"summary_refL{ref_layer}.json", "w") as f:
            json.dump({
                "ref_layer": int(ref_layer),
                "masses": mass_list,
                "metrics_by_mass": metrics_by_mass,
            }, f, indent=2)

        all_metrics[str(ref_layer)] = metrics_by_mass
        plot_data[ref_layer] = (masses_for_plot, mae_points)

    # Save aggregate summary JSON across refs
    with open(out_dir / "summary_all_refs.json", "w") as f:
        json.dump({
            "ref_layers": ref_layers,
            "masses": mass_list,
            "by_ref_layer": all_metrics,
        }, f, indent=2)

    # Plot world_points_mae vs mass for all ref layers
    if any(len(v[0]) > 0 for v in plot_data.values()):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for ref_layer, (masses_for_plot, mae_points) in sorted(plot_data.items()):
            if len(masses_for_plot) == 0:
                continue
            ax.plot(masses_for_plot, mae_points, marker="o", label=f"ref L{ref_layer}")
        ax.set_xlabel("Mass preserved")
        ax.set_ylabel("World points MAE")
        ax.set_title("World points MAE vs mass (multiple ref layers)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        refs_tag = "-".join([f"L{r}" for r in sorted(ref_layers)])
        plot_path = out_dir / f"world_points_mae_vs_mass_refs_{refs_tag}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()


