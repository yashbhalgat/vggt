import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_metadata(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def _discover_layers(path: Path) -> List[Tuple[int, Path]]:
    """
    Accepts either the attn_maps directory, the layers_manifest.json, or a single layer metadata file.
    Returns a sorted list of (layer_index, metadata_path).
    """
    if path.is_dir():
        manifest = path / "layers_manifest.json"
        if manifest.exists():
            data = _load_metadata(manifest)
            layers = []
            for l in data.get("layers", []):
                idx = int(l.get("index"))
                mp_raw = Path(l.get("metadata_path"))
                mp = mp_raw if mp_raw.is_absolute() else manifest.parent / mp_raw
                if not mp.exists():
                    # try basename fallback
                    alt = manifest.parent / mp_raw.name
                    mp = alt if alt.exists() else mp
                layers.append((idx, mp))
            layers.sort(key=lambda x: x[0])
            return layers
        # fallback: glob layer_*_metadata.json
        items: List[Tuple[int, Path]] = []
        for mp in path.glob("layer_*_metadata.json"):
            name = mp.stem
            try:
                idx = int(name.split("_")[1]) - 1
                items.append((idx, mp))
            except Exception:
                continue
        items.sort(key=lambda x: x[0])
        return items
    else:
        if path.name == "layers_manifest.json":
            return _discover_layers(path.parent)
        # single metadata
        name = path.stem
        try:
            idx = int(name.split("_")[1]) - 1
        except Exception:
            idx = 0
        return [(idx, path)]


def _resolve_npz_path(meta: Dict[str, Any], meta_path: Path) -> Path:
    raw = meta.get("npz_path") or meta.get("npy_path")
    if raw is None:
        raise RuntimeError(f"No npz_path/npy_path in metadata: {meta_path}")
    p = Path(raw)
    return p if p.is_absolute() else meta_path.parent / p


def _load_layer_array(meta_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load layer heatmaps into shape [Q, S, H, W] as float32.
    If per_head_data, average across heads.
    """
    meta = _load_metadata(meta_path)
    npz_path = _resolve_npz_path(meta, meta_path)
    loaded = np.load(npz_path, mmap_mode="r")
    arr = loaded["heatmaps"] if hasattr(loaded, "files") else loaded

    # Convert and reduce heads if present
    if meta.get("per_head_data", False) and arr.ndim == 5:
        # [Q, H, S, Hq, Wq] -> average over head axis 1
        arr = arr.astype(np.float32).mean(axis=1)
    else:
        arr = arr.astype(np.float32)

    return arr, meta


def _masses_default() -> List[float]:
    # Default mass fractions for five stacked bar plots
    return [0.3, 0.4, 0.5, 0.6, 0.7]


def _to_mask_mass(arr: np.ndarray, mass_fraction: float) -> np.ndarray:
    """
    Threshold arr to preserve a given fraction of total mass.
    Select the largest values until cumulative sum >= mass_fraction * total_sum.
    Returns boolean mask of the same shape.
    """
    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool)
    frac = float(np.clip(mass_fraction, 0.0, 1.0))
    if frac <= 0.0:
        return np.zeros_like(arr, dtype=bool)
    flat = arr.reshape(-1).astype(np.float64)
    total = flat.sum()
    if total <= 0.0:
        return np.zeros_like(arr, dtype=bool)
    target = frac * total
    # Sort descending to find cutoff efficiently
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    cutoff_idx = int(np.searchsorted(csum, target, side="left"))
    if cutoff_idx >= flat.size:
        # everything selected
        return np.ones_like(arr, dtype=bool)
    threshold = flat[order[cutoff_idx]]
    # Include ties at threshold
    return arr >= threshold


def _recall_tnr(mask_ref: np.ndarray, mask_next: np.ndarray) -> Tuple[float, float]:
    """
    Treat mask_ref as reference labels, mask_next as prediction.
    Return (recall, true_negative_rate).
    """
    ref = mask_ref.reshape(-1)
    pred = mask_next.reshape(-1)
    tp = np.logical_and(ref, pred).sum(dtype=np.int64)
    fn = np.logical_and(ref, np.logical_not(pred)).sum(dtype=np.int64)
    tn = np.logical_and(np.logical_not(ref), np.logical_not(pred)).sum(dtype=np.int64)
    fp = np.logical_and(np.logical_not(ref), pred).sum(dtype=np.int64)

    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0
    tnr = float(tn) / float(tn + fp) if (tn + fp) > 0 else 1.0
    return recall, tnr


def analyze(attn_path: Path, start_layer: int, out_dir: Path, masses: List[float]) -> None:
    # Discover layers and load arrays
    layers = _discover_layers(attn_path)
    if len(layers) < 2:
        raise RuntimeError("Need at least two layers for analysis")

    # Load all required layers into memory (float32)
    arrays: Dict[int, np.ndarray] = {}
    metas: Dict[int, Dict[str, Any]] = {}
    for li, meta_path in layers:
        arr, meta = _load_layer_array(meta_path)
        arrays[li] = arr
        metas[li] = meta

    layer_indices = sorted(arrays.keys())
    # Ensure shapes are consistent across layers
    base_shape = arrays[layer_indices[0]].shape
    for li in layer_indices[1:]:
        if arrays[li].shape != base_shape:
            raise RuntimeError(f"Layer {li} has shape {arrays[li].shape}, expected {base_shape}")

    # Determine reference and target layers (j > start_layer)
    if start_layer not in arrays:
        raise RuntimeError(f"Start layer {start_layer} not found in attn maps")
    layer_js = [li for li in layer_indices if li > start_layer]
    if len(layer_js) == 0:
        raise RuntimeError("No layers found after start_layer for comparison")

    # Compute metrics vs fixed reference
    ref_arr = arrays[start_layer]
    results_by_mass: Dict[float, List[Tuple[int, float, float]]] = {}
    per_layer_rows = []  # detailed rows for CSV

    for mass in masses:
        m_ref = _to_mask_mass(ref_arr, mass)
        rows_mass: List[Tuple[int, float, float]] = []
        for lj in layer_js:
            nxt = arrays[lj]
            m_nxt = _to_mask_mass(nxt, mass)
            r, t = _recall_tnr(m_ref, m_nxt)
            rows_mass.append((lj, r, t))
            per_layer_rows.append({
                "layer_ref": start_layer,
                "layer_j": lj,
                "mass_fraction": mass,
                "recall": r,
                "tnr": t,
            })
        results_by_mass[mass] = rows_mass

    # Save CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"attention_stability_refL{start_layer}_mass_detailed.csv"
    with open(csv_path, "w") as f:
        f.write("layer_ref,layer_j,mass_fraction,recall,tnr\n")
        for row in per_layer_rows:
            f.write(f"{row['layer_ref']},{row['layer_j']},{row['mass_fraction']:.4f},{row['recall']:.6f},{row['tnr']:.6f}\n")

    # Stacked bar plots: one subplot per percentile
    nrows = len(masses)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 2.6 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    x = np.arange(len(layer_js))
    width = 0.35
    for i, mass in enumerate(masses):
        ax = axes[i]
        data = results_by_mass[mass]
        # Ensure alignment with layer_js
        layer_to_vals = {lj: (r, t) for lj, r, t in data}
        recalls = [layer_to_vals[lj][0] for lj in layer_js]
        tnrs = [layer_to_vals[lj][1] for lj in layer_js]

        ax.bar(x - width / 2, recalls, width, label="Recall", color="#1f77b4")
        ax.bar(x + width / 2, tnrs, width, label="TNR", color="#ff7f0e")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.set_ylabel("Score")
        ax.set_title(f"Mass preserved: {int(round(mass * 100))}%")
        if i == 0:
            ax.legend(ncols=2)

    labels = [f"layer {lj}" for lj in layer_js]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel(f"Layer j (ref = {start_layer})")
    fig.suptitle(f"Attention stability vs reference layer {start_layer}", fontsize=30)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = out_dir / f"attention_stability_refL{start_layer}_bars_mass.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot: {plot_path}")
    print(f"Saved CSV:  {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_path", type=str, required=True, help="Path to attn_maps directory or layers_manifest.json")
    parser.add_argument("--start_layer", type=int, default=12, help="0-indexed layer to start comparisons from")
    parser.add_argument("--out_dir", type=str, default="attn_analysis", help="Output directory for plots and CSV")
    parser.add_argument("--masses", type=str, default="0.3,0.4,0.5,0.6,0.7", help="Comma-separated mass fractions (e.g., 0.3,0.4,0.5,0.6,0.7)")
    args = parser.parse_args()

    attn_path = Path(args.attn_path)
    out_dir = Path(args.out_dir)
    try:
        masses = [float(x.strip()) for x in args.masses.split(",") if x.strip()]
    except Exception:
        masses = _masses_default()
    analyze(attn_path, args.start_layer, out_dir, masses)


if __name__ == "__main__":
    main()


