import argparse
import os
from pathlib import Path
import shutil
import json
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from matplotlib import cm
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.attn_capture import FirstToAllAttentionCapture

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load_model():
    # model loading
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model = model.to(device)
    return model

def load_images(image_names):
    images = load_and_preprocess_images(image_names).to(device)
    return images

def _save_layer_npz(
    grids: List[np.ndarray],
    image_paths: List[Path],
    out_root: Path,
    stride_h: int,
    stride_w: int,
    H_p: int,
    W_p: int,
) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)

    # Load image sizes and ensure consistent size across frames
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    sizes = [(im.width, im.height) for im in pil_images]
    if len(set(sizes)) != 1:
        raise RuntimeError("All images must have the same size for NPZ export")
    W_img, H_img = sizes[0]
    img_sizes = [{"width": W_img, "height": H_img} for _ in pil_images]

    # Determine downsampled grid size from one grid
    S, H_ds, W_ds = grids[0].shape if len(grids) > 0 else (len(image_paths), 0, 0)

    # Compute layerwise vmax (99th percentile) across all grids
    def quantile_99(a: np.ndarray) -> float:
        if a.size == 0:
            return 1.0
        return float(np.quantile(a, 0.99))

    if len(grids) > 0:
        flat_all = np.concatenate([g.reshape(-1) for g in grids], axis=0)
        layer_vmax99 = quantile_99(flat_all)
    else:
        layer_vmax99 = 1.0

    # Stack to [Q, S, H_ds, W_ds]
    Q = len(grids)
    stacked = np.stack(grids, axis=0).astype(np.float32)
    # Normalize by layerwise vmax
    stacked = np.clip(stacked / (layer_vmax99 + 1e-8), 0.0, 1.0)

    # Quantize at low-res grid (H_ds x W_ds); no upsampling here
    heatmaps_u8 = (stacked * 255.0 + 0.5).astype(np.uint8)  # [Q,S,H_ds,W_ds]

    # Save single compressed NPZ
    npz_path = out_root / "heatmaps_uint8.npz"
    np.savez_compressed(npz_path, heatmaps=heatmaps_u8)

    # Build metadata
    metadata = {
        "S": int(len(image_paths)),
        "H_p": int(H_p),
        "W_p": int(W_p),
        "stride_h": int(stride_h),
        "stride_w": int(stride_w),
        "H_ds": int(H_ds),
        "W_ds": int(W_ds),
        "layer_vmax99": float(layer_vmax99),
        "normalized_by": "layer_vmax99",
        # store npz path relative to attn_maps dir to avoid double prefixing when loading
        "npz_path": os.path.relpath(str(npz_path), start=str(out_root.parent)),
        # store paths relative to attn_maps dir, even if images are in a sibling dir
        "image_paths": [os.path.relpath(str(p), start=str(out_root.parent)) for p in image_paths],
        "image_sizes": img_sizes,
        # Query grid indexing info (qr, qc) can be derived on the fly
    }
    return metadata


def compute_layerwise_attentions(
    model,
    images,
    image_set_name: str = "kitchen",
    output_dir: str = "attn_outputs_first2all",
    stride_h: int = 1,
    stride_w: int = 1,
    max_query_chunk: int = 2048,
):
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
        if dtype is not None and images.is_cuda:
            with torch.cuda.amp.autocast(dtype=dtype):
                _ = model(images)
        else:
            _ = model(images)

    capturer.clear()

    # We captured all layers in a single forward pass
    depth = len(getattr(model.aggregator, "global_blocks", []))
    if depth == 0:
        raise RuntimeError("No global_blocks found on model.aggregator")

    # Gather patch grid metadata from model context
    ctx = getattr(getattr(model, "aggregator", object), "_attn_context", None)
    if ctx is None:
        raise RuntimeError("Missing _attn_context on model.aggregator; cannot determine patch grid size")
    H_p, W_p = ctx.get("HW_patches", (0, 0))

    # Determine the copied image paths in output_dir/images
    images_dir = Path(output_dir) / "images"
    image_paths = [p for p in sorted(images_dir.glob("*")) if p.is_file()]
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    # Save heatmaps and metadata for all layers; build manifest
    attn_root = Path(output_dir) / "attn_maps"
    attn_root.mkdir(parents=True, exist_ok=True)
    manifest = {"image_set_name": image_set_name, "layers": []}

    for li in sorted(capturer._layer_grids.keys()):
        grids = capturer._layer_grids[li]  # List[np.ndarray] length Q, each [S, H_ds, W_ds]
        layer_dir = attn_root / f"layer_{li + 1:02d}"
        metadata = _save_layer_npz(
            grids=grids,
            image_paths=image_paths,
            out_root=layer_dir,
            stride_h=stride_h,
            stride_w=stride_w,
            H_p=H_p,
            W_p=W_p,
        )
        metadata_path = attn_root / f"layer_{li + 1:02d}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        manifest["layers"].append(
            {
                "index": int(li),
                "name": f"layer_{li + 1:02d}",
                "metadata_path": str(metadata_path),
                "layer_vmax99": metadata.get("layer_vmax99", 1.0),
            }
        )

    manifest_path = attn_root / "layers_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if was_training:
        model.train()

    return str(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_set_name", type=str, default="kitchen")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--output_dir_prefix", type=str, default="attn_first2all_")
    parser.add_argument("--stride_h", type=int, default=1)
    parser.add_argument("--stride_w", type=int, default=1)
    args = parser.parse_args()
    
    output_dir = args.output_dir_prefix + args.image_set_name
    
    model = load_model()
    
    image_dir = Path(f"examples/{args.image_set_name}/images")
    all_image_names = [f for f in image_dir.glob("*.png")]
    step = len(all_image_names) // args.num_images
    # take every step-th image
    image_names = [str(all_image_names[i]) for i in range(0, len(all_image_names), step)]
    images = load_images(image_names)
    
    # copy images to output_dir / images
    os.makedirs(Path(output_dir) / "images", exist_ok=True)
    for image_name in image_names:
        shutil.copy(image_name, Path(output_dir) / "images" / os.path.basename(image_name))
    
    manifest_path = compute_layerwise_attentions(
        model,
        images,
        image_set_name=args.image_set_name,
        output_dir=output_dir,
        stride_h=args.stride_h,
        stride_w=args.stride_w,
        max_query_chunk=2048,
    )
    print(manifest_path)
    