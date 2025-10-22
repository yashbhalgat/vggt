import math
import os
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt


class GlobalAttentionCapture:
    def __init__(self, model, image_set_name: str, output_dir: str = "attn_outputs") -> None:
        self.model = model
        self.image_set_name = image_set_name
        self.output_dir = output_dir
        self._handles: List = []
        self._layer_maps: Dict[int, np.ndarray] = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def _make_callback(self, layer_index: int):
        def callback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, module) -> None:
            # q, k, v: [B, H, N, D]
            aggregator = getattr(self.model, "aggregator", None)
            if aggregator is None or not hasattr(aggregator, "_attn_context"):
                return

            ctx = aggregator._attn_context
            B = int(ctx.get("B", 0))
            S = int(ctx.get("S", 0))
            P_frame = int(ctx.get("P_frame", 0))
            patch_start_idx = int(ctx.get("patch_start_idx", 0))
            H_p, W_p = ctx.get("HW_patches", (0, 0))

            if B <= 0 or S <= 0 or P_frame <= 0 or H_p == 0 or W_p == 0:
                return

            # Indices for each frame's camera token within the flattened global sequence
            device = q.device
            head_dim = q.shape[-1]
            camera_indices = torch.arange(S, device=device) * P_frame

            # Select queries at camera tokens: [B, H, S, D]
            q_cam = q[:, :, camera_indices, :].float()
            k_all = k.float()  # [B, H, N, D]

            # Compute scaled dot products only for camera queries against all keys
            # logits: [B, H, S, N]
            logits = torch.einsum("bhsd,bhnd->bhsn", q_cam, k_all) / math.sqrt(head_dim)
            attn = torch.softmax(logits, dim=-1)

            # For each frame, extract attention to that frame's patch tokens, then mean over heads
            maps_list: List[torch.Tensor] = []
            for s in range(S):
                start = s * P_frame + patch_start_idx
                end = (s + 1) * P_frame
                # [B, H, num_patches]
                attn_s = attn[:, :, s, start:end]
                # [B, num_patches]
                attn_s_mean = attn_s.mean(dim=1)
                # [B, H_p, W_p]
                maps_list.append(attn_s_mean.reshape(B, H_p, W_p))

            # Stack across frames => [B, S, H_p, W_p] and average batch if B>1
            maps = torch.stack(maps_list, dim=1)
            maps = maps.mean(dim=0)  # [S, H_p, W_p]

            # Store raw softmax probabilities (no normalization)
            maps_np = maps.detach().cpu().numpy()
            self._layer_maps[layer_index] = maps_np

        return callback

    def _make_pre_hook(self, layer_index: int):
        def hook(module, args, kwargs):
            # args: (x,), kwargs may contain 'pos'
            x = args[0]
            pos = None
            if isinstance(kwargs, dict):
                pos = kwargs.get("pos", None)

            # Compute q, k, v like in Attention.forward
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            if getattr(module, "rope", None) is not None:
                q = module.rope(q, pos)
                k = module.rope(k, pos)

            callback = self._make_callback(layer_index)
            callback(q, k, v, module)

        return hook

    def register(self) -> None:
        # Attach forward pre-hooks to all global attention layers
        blocks = getattr(self.model.aggregator, "global_blocks", [])
        self._handles = []
        for i, block in enumerate(blocks):
            if hasattr(block, "attn"):
                try:
                    handle = block.attn.register_forward_pre_hook(self._make_pre_hook(i), with_kwargs=True)
                except TypeError:
                    handle = block.attn.register_forward_pre_hook(self._make_pre_hook(i))
                self._handles.append(handle)

    def clear(self) -> None:
        # Remove hooks
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def save_figures(self) -> List[str]:
        saved_paths: List[str] = []
        depth = len(getattr(self.model.aggregator, "global_blocks", []))
        for layer_idx in range(depth):
            if layer_idx not in self._layer_maps:
                continue
            maps = self._layer_maps[layer_idx]  # [S, H_p, W_p]
            S = maps.shape[0]
            ncols = min(5, S)
            nrows = int(math.ceil(S / ncols))

            # Layerwise vmax using 99th percentile
            layer_vmax = float(np.quantile(maps, 0.99)) if maps.size > 0 else 1.0

            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
            axes = np.atleast_2d(axes)
            for s in range(nrows * ncols):
                r, c = divmod(s, ncols)
                ax = axes[r, c]
                ax.axis("off")
                if s < S:
                    ax.imshow(maps[s], cmap="viridis", vmin=0.0, vmax=layer_vmax)
                    ax.set_title(f"frame {s}")

            fig.suptitle(
                f"{self.image_set_name} — global attention layer {layer_idx + 1}  (vmax={layer_vmax:.4f})",
                fontsize=14,
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            out_path = os.path.join(
                self.output_dir, f"{self.image_set_name}_global_attn_layer_{layer_idx + 1:02d}.png"
            )
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved_paths.append(out_path)
        return saved_paths


def capture_and_save_global_attention(
    model,
    images: torch.Tensor,
    image_set_name: str,
    output_dir: str = "attn_outputs",
    amp_dtype: torch.dtype = None,
) -> List[str]:
    """
    Runs the model once with callbacks attached to global attention layers, collects
    head-averaged camera-token-to-patches attention per frame for each layer, and
    saves per-layer figures to disk.
    """
    was_training = model.training
    model.eval()

    capturer = GlobalAttentionCapture(model, image_set_name=image_set_name, output_dir=output_dir)
    capturer.register()

    with torch.no_grad():
        if amp_dtype is not None and images.is_cuda:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                _ = model(images)
        else:
            _ = model(images)

    capturer.clear()
    saved = capturer.save_figures()

    if was_training:
        model.train()

    return saved


class PatchGridAttentionCapture:
    def __init__(
        self,
        model,
        image_set_name: str,
        output_dir: str = "attn_outputs",
        stride_h: int = 1,
        stride_w: int = 1,
        max_query_chunk: int = 2048,
    ) -> None:
        self.model = model
        self.image_set_name = image_set_name
        self.output_dir = output_dir
        self.stride_h = max(1, int(stride_h))
        self.stride_w = max(1, int(stride_w))
        self.max_query_chunk = max(1, int(max_query_chunk))

        self._handles: List = []
        self._layer_matrix: Dict[int, np.ndarray] = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def _compute_patch_indices(self, S: int, P_frame: int, patch_start_idx: int, H_p: int, W_p: int) -> Tuple[torch.Tensor, int]:
        H_ds = (H_p + self.stride_h - 1) // self.stride_h
        W_ds = (W_p + self.stride_w - 1) // self.stride_w
        per_frame = H_ds * W_ds
        # Build indices for downsampled grid within a frame
        grid_idx = []
        for r in range(0, H_p, self.stride_h):
            for c in range(0, W_p, self.stride_w):
                linear = r * W_p + c
                grid_idx.append(linear)
        frame_patch_idx = torch.tensor(grid_idx, dtype=torch.long)

        # Offset to absolute token positions in the global sequence per frame
        all_idx = []
        for s in range(S):
            base = s * P_frame + patch_start_idx
            all_idx.append(base + frame_patch_idx)
        all_idx = torch.cat(all_idx, dim=0)  # [S*per_frame]
        return all_idx, per_frame

    def _make_pre_hook(self, layer_index: int):
        def hook(module, args, kwargs):
            x = args[0]
            pos = kwargs.get("pos", None) if isinstance(kwargs, dict) else None

            aggregator = getattr(self.model, "aggregator", None)
            if aggregator is None or not hasattr(aggregator, "_attn_context"):
                return

            ctx = aggregator._attn_context
            B = int(ctx.get("B", 0))
            S = int(ctx.get("S", 0))
            P_frame = int(ctx.get("P_frame", 0))
            patch_start_idx = int(ctx.get("patch_start_idx", 0))
            H_p, W_p = ctx.get("HW_patches", (0, 0))
            if B <= 0 or S <= 0 or P_frame <= 0 or H_p == 0 or W_p == 0:
                return

            # Build patch-only indices (downsampled)
            patch_indices, per_frame = self._compute_patch_indices(S, P_frame, patch_start_idx, H_p, W_p)
            patch_indices = patch_indices.to(x.device)
            Np = patch_indices.numel()

            # Compute q, k for those positions
            Bx, N, C = x.shape
            qkv = module.qkv(x).reshape(Bx, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            if getattr(module, "rope", None) is not None:
                q = module.rope(q, pos)
                k = module.rope(k, pos)

            q_sel = q.index_select(dim=2, index=patch_indices).float()  # [B, H, Np, D]
            k_sel = k.index_select(dim=2, index=patch_indices).float()  # [B, H, Np, D]

            head_dim = q_sel.shape[-1]
            # Allocate output matrix incrementally to avoid peak memory
            attn_accum = torch.zeros((Np, Np), dtype=torch.float32, device=q_sel.device)
            count_heads = q_sel.shape[0] * q_sel.shape[1]

            # Chunk over query dimension
            for start in range(0, Np, self.max_query_chunk):
                stop = min(Np, start + self.max_query_chunk)
                # [B, H, chunk, D] x [B, H, Np, D]^T -> [B, H, chunk, Np]
                logits = torch.einsum("bhqd,bhkd->bhqk", q_sel[:, :, start:stop, :], k_sel) / math.sqrt(head_dim)
                probs = torch.softmax(logits, dim=-1)
                # Mean over batch and heads -> [chunk, Np]
                probs_mean = probs.mean(dim=(0, 1))
                attn_accum[start:stop, :] = probs_mean.to(dtype=torch.float32)

            # Move to CPU numpy
            mat = attn_accum.detach().cpu().numpy()
            self._layer_matrix[layer_index] = mat

        return hook

    def register(self) -> None:
        blocks = getattr(self.model.aggregator, "global_blocks", [])
        self._handles = []
        for i, block in enumerate(blocks):
            if hasattr(block, "attn"):
                try:
                    handle = block.attn.register_forward_pre_hook(self._make_pre_hook(i), with_kwargs=True)
                except TypeError:
                    handle = block.attn.register_forward_pre_hook(self._make_pre_hook(i))
                self._handles.append(handle)

    def clear(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def save_figures(self) -> List[str]:
        saved_paths: List[str] = []
        aggregator = getattr(self.model, "aggregator", None)
        if aggregator is None or not hasattr(aggregator, "_attn_context"):
            return saved_paths
        ctx = aggregator._attn_context
        S = int(ctx.get("S", 0))
        H_p, W_p = ctx.get("HW_patches", (0, 0))
        _, per_frame = self._compute_patch_indices(S, int(ctx.get("P_frame", 0)), int(ctx.get("patch_start_idx", 0)), H_p, W_p)

        depth = len(getattr(self.model.aggregator, "global_blocks", []))
        for layer_idx in range(depth):
            if layer_idx not in self._layer_matrix:
                continue
            mat = self._layer_matrix[layer_idx]

            # Plot matrix with frame separators
            Np = mat.shape[0]
            n_segs = S
            fig_size = max(6, min(16, Np / 200))
            fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
            layer_vmax = float(np.quantile(mat, 0.99)) if mat.size > 0 else 1.0
            im = ax.imshow(mat, cmap="magma", interpolation="nearest", vmin=0.0, vmax=layer_vmax)
            ax.set_title(f"{self.image_set_name} — global patch-to-patch layer {layer_idx + 1}\nstride=({self.stride_h},{self.stride_w})")
            # Frame separators
            for t in range(1, n_segs):
                pos = t * per_frame
                ax.axhline(pos - 0.5, color="white", lw=0.5, alpha=0.5)
                ax.axvline(pos - 0.5, color="white", lw=0.5, alpha=0.5)
            ax.set_xlabel("keys (frames concatenated)")
            ax.set_ylabel("queries (frames concatenated)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            out_path = os.path.join(
                self.output_dir,
                f"{self.image_set_name}_global_patchgrid_layer_{layer_idx + 1:02d}_s{self.stride_h}x{self.stride_w}.png",
            )
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved_paths.append(out_path)
        return saved_paths


def capture_and_save_patch_grid_attention(
    model,
    images: torch.Tensor,
    image_set_name: str,
    output_dir: str = "attn_outputs",
    stride_h: int = 1,
    stride_w: int = 1,
    amp_dtype: Optional[torch.dtype] = None,
    max_query_chunk: int = 2048,
) -> List[str]:
    was_training = model.training
    model.eval()

    capturer = PatchGridAttentionCapture(
        model,
        image_set_name=image_set_name,
        output_dir=output_dir,
        stride_h=stride_h,
        stride_w=stride_w,
        max_query_chunk=max_query_chunk,
    )
    capturer.register()

    with torch.no_grad():
        if amp_dtype is not None and images.is_cuda:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                _ = model(images)
        else:
            _ = model(images)

    capturer.clear()
    saved = capturer.save_figures()

    if was_training:
        model.train()

    return saved


class FirstToAllAttentionCapture:
    def __init__(
        self,
        model,
        image_set_name: str,
        output_dir: str = "attn_outputs",
        stride_h: int = 1,
        stride_w: int = 1,
        max_query_chunk: int = 2048,
    ) -> None:
        self.model = model
        self.image_set_name = image_set_name
        self.output_dir = output_dir
        self.stride_h = max(1, int(stride_h))
        self.stride_w = max(1, int(stride_w))
        self.max_query_chunk = max(1, int(max_query_chunk))

        self._handles: List = []
        # For each layer we will store a list of (H' * W') heatmaps, each shaped [S, H', W']
        self._layer_grids: Dict[int, List[np.ndarray]] = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def _downsample_indices(self, H_p: int, W_p: int) -> Tuple[torch.Tensor, int, int]:
        H_ds = (H_p + self.stride_h - 1) // self.stride_h
        W_ds = (W_p + self.stride_w - 1) // self.stride_w
        grid_idx = []
        for r in range(0, H_p, self.stride_h):
            for c in range(0, W_p, self.stride_w):
                grid_idx.append(r * W_p + c)
        return torch.tensor(grid_idx, dtype=torch.long), H_ds, W_ds

    def _make_pre_hook(self, layer_index: int):
        def hook(module, args, kwargs):
            x = args[0]
            pos = kwargs.get("pos", None) if isinstance(kwargs, dict) else None

            aggregator = getattr(self.model, "aggregator", None)
            if aggregator is None or not hasattr(aggregator, "_attn_context"):
                return

            ctx = aggregator._attn_context
            B = int(ctx.get("B", 0))
            S = int(ctx.get("S", 0))
            P_frame = int(ctx.get("P_frame", 0))
            patch_start_idx = int(ctx.get("patch_start_idx", 0))
            H_p, W_p = ctx.get("HW_patches", (0, 0))
            if B <= 0 or S <= 0 or P_frame <= 0 or H_p == 0 or W_p == 0:
                return

            patch_idx_within, H_ds, W_ds = self._downsample_indices(H_p, W_p)
            patch_idx_within = patch_idx_within.to(x.device)
            per_frame = patch_idx_within.numel()

            # Absolute indices for keys: all frames' patch tokens
            key_indices = []
            for s in range(S):
                base = s * P_frame + patch_start_idx
                key_indices.append(base + patch_idx_within)
            key_indices = torch.cat(key_indices, dim=0)  # [S*per_frame]

            # Absolute indices for queries: ONLY frame 0 patch tokens
            query_indices = patch_idx_within + patch_start_idx  # [per_frame]

            # Compute q, k
            Bx, N, C = x.shape
            qkv = module.qkv(x).reshape(Bx, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            if getattr(module, "rope", None) is not None:
                q = module.rope(q, pos)
                k = module.rope(k, pos)

            q_sel = q.index_select(dim=2, index=query_indices).float()   # [B, H, Q, D]
            k_sel = k.index_select(dim=2, index=key_indices).float()     # [B, H, K, D]

            head_dim = q_sel.shape[-1]
            Q = q_sel.shape[2]
            K = k_sel.shape[2]

            # We will build a per-query map shaped [S, H_ds, W_ds] for each query in frame 0
            grids: List[np.ndarray] = []

            for start in range(0, Q, self.max_query_chunk):
                stop = min(Q, start + self.max_query_chunk)
                # logits: [B, H, chunk, K]
                logits = torch.einsum("bhqd,bhkd->bhqk", q_sel[:, :, start:stop, :], k_sel) / math.sqrt(head_dim)
                probs = torch.softmax(logits, dim=-1)
                # mean over batch and heads -> [chunk, K]
                probs_mean = probs.mean(dim=(0, 1))
                # Split K back into S blocks of size per_frame and reshape to [S, H_ds, W_ds]
                probs_blocks = probs_mean.split(per_frame, dim=-1)
                for q_local in range(probs_blocks[0].shape[0]):
                    # q_local iterates per query inside the current chunk
                    flat = torch.stack([b[q_local] for b in probs_blocks], dim=0)  # [S, per_frame]
                    grid = flat.reshape(S, H_ds, W_ds).detach().cpu().numpy()
                    grids.append(grid)

            self._layer_grids[layer_index] = grids  # length Q = H_ds*W_ds

        return hook

    def register(self) -> None:
        blocks = getattr(self.model.aggregator, "global_blocks", [])
        self._handles = []
        for i, block in enumerate(blocks):
            if hasattr(block, "attn"):
                try:
                    handle = block.attn.register_forward_pre_hook(self._make_pre_hook(i), with_kwargs=True)
                except TypeError:
                    handle = block.attn.register_forward_pre_hook(self._make_pre_hook(i))
                self._handles.append(handle)

    def clear(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def save_figures(self) -> List[str]:
        saved_paths: List[str] = []
        depth = len(getattr(self.model.aggregator, "global_blocks", []))
        for layer_idx in range(depth):
            if layer_idx not in self._layer_grids:
                continue
            grids = self._layer_grids[layer_idx]  # len Q = H_ds*W_ds, each [S, H_ds, W_ds]
            if len(grids) == 0:
                continue

            S, H_ds, W_ds = grids[0].shape
            K_cols = int(np.ceil(np.sqrt(S)))
            K_rows = int(np.ceil(S / K_cols))
            pad = K_rows * K_cols - S

            # Build (K_rows x K_cols) composite per query
            composed = []
            for grid in grids:
                if pad > 0:
                    pad_block = np.zeros((pad, H_ds, W_ds), dtype=grid.dtype)
                    grid = np.concatenate([grid, pad_block], axis=0)
                tiles = []
                for r in range(K_rows):
                    row_tiles = []
                    for c in range(K_cols):
                        s = r * K_cols + c
                        row_tiles.append(grid[s])
                    tiles.append(np.concatenate(row_tiles, axis=1))
                big = np.concatenate(tiles, axis=0)
                composed.append(big)

            # Arrange as H_ds x W_ds (query layout)
            fig_h = max(6, H_ds * 1.2)
            fig_w = max(8, W_ds * 1.2)
            fig, axes = plt.subplots(H_ds, W_ds, figsize=(fig_w, fig_h))
            axes = np.atleast_2d(axes)

            for qr in range(H_ds):
                for qc in range(W_ds):
                    qi = qr * W_ds + qc
                    ax = axes[qr, qc]
                    if qi < len(composed):
                        img = composed[qi]
                        # Layerwise vmax across all tiles in this layer using 99th percentile
                        layer_vmax = float(np.quantile(composed, 0.99)) if len(composed) > 0 else 1.0
                        ax.imshow(img, cmap="viridis", interpolation="nearest", vmin=0.0, vmax=layer_vmax)
                        ax.axis("off")
                        for y in range(1, K_rows):
                            ax.axhline(y * H_ds - 0.5, color="white", lw=0.4, alpha=0.6)
                        for x in range(1, K_cols):
                            ax.axvline(x * W_ds - 0.5, color="white", lw=0.4, alpha=0.6)
                        # annotate frames
                        for r in range(K_rows):
                            for c in range(K_cols):
                                s = r * K_cols + c
                                if s < S:
                                    ax.text(c * W_ds + W_ds * 0.1, r * H_ds + H_ds * 0.2, f"f{s}", color="white", fontsize=4, ha="left", va="top", alpha=0.8)
                    else:
                        ax.axis("off")

            layer_vmax_all = float(np.quantile(composed, 0.99)) if len(composed) > 0 else 1.0
            fig.suptitle(
                f"{self.image_set_name} — first-to-all (frame 0 queries) layer {layer_idx + 1} | tiles={K_rows}x{K_cols}, query grid={H_ds}x{W_ds}, stride=({self.stride_h},{self.stride_w})\n"
                f"vmax={layer_vmax_all:.4f}",
                fontsize=14,
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            out_path = os.path.join(
                self.output_dir,
                f"{self.image_set_name}_first2all_layer_{layer_idx + 1:02d}_T{K_rows}x{K_cols}_q{H_ds}x{W_ds}_s{self.stride_h}x{self.stride_w}.png",
            )
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved_paths.append(out_path)

        return saved_paths


def capture_and_save_first_to_all_attention(
    model,
    images: torch.Tensor,
    image_set_name: str,
    output_dir: str = "attn_outputs",
    stride_h: int = 1,
    stride_w: int = 1,
    amp_dtype: Optional[torch.dtype] = None,
    max_query_chunk: int = 2048,
) -> List[str]:
    was_training = model.training
    model.eval()

    capturer = FirstToAllAttentionCapture(
        model,
        image_set_name=image_set_name,
        output_dir=output_dir,
        stride_h=stride_h,
        stride_w=stride_w,
        max_query_chunk=max_query_chunk,
    )
    capturer.register()

    with torch.no_grad():
        if amp_dtype is not None and images.is_cuda:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                _ = model(images)
        else:
            _ = model(images)

    capturer.clear()
    saved = capturer.save_figures()

    if was_training:
        model.train()

    return saved

