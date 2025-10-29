import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
from matplotlib import cm


def _load_metadata(metadata_path: Path) -> Dict[str, Any]:
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    return meta


def _resolve_under(base: Path, p: Path) -> Path:
    # Try absolute
    if p.is_absolute():
        return p
    # Try base / p
    candidate = base / p
    if candidate.exists():
        return candidate
    # If p redundantly includes base dir name, strip leading duplicates
    parts = list(p.parts)
    while parts and (parts[0] == base.name or parts[0] == base.parent.name):
        parts = parts[1:]
        candidate2 = base / Path(*parts)
        if candidate2.exists():
            return candidate2
    # Fallback to basename in base
    candidate3 = base / p.name
    if candidate3.exists():
        return candidate3
    return candidate


def _resolve_paths(meta: Dict[str, Any], metadata_path: Path) -> Tuple[List[Path], Dict[str, Any]]:
    root = metadata_path.parent
    image_paths = [_resolve_under(root, Path(p)) for p in meta["image_paths"]]
    if "npz_path" in meta or "npy_path" in meta:
        p = meta.get("npz_path") or meta.get("npy_path")
        p = _resolve_under(root, Path(p))
        return image_paths, {"mode": "npz", "path": p}
    else:
        # backward-compatible PNG list
        all_heatmaps: List[List[Path]] = []
        for q in meta["queries"]:
            hm = [_resolve_under(root, Path(p)) for p in q["heatmaps"]]
            all_heatmaps.append(hm)
        return image_paths, {"mode": "png", "paths": all_heatmaps}


def _discover_layers(path: Path) -> Tuple[List[str], Dict[str, Path]]:
    # Accept directory (attn_maps) or manifest json or single layer metadata json
    layer_map: Dict[str, Path] = {}
    if path.is_dir():
        # look for manifest first
        manifest = path / "layers_manifest.json"
        if manifest.exists():
            data = _load_metadata(manifest)
            for l in data.get("layers", []):
                name = l.get("name")
                mp_raw = Path(l.get("metadata_path"))
                # Resolve metadata path robustly
                if mp_raw.is_absolute():
                    mp_resolved = mp_raw
                else:
                    candidate = manifest.parent / mp_raw
                    if candidate.exists():
                        mp_resolved = candidate
                    else:
                        # Fallback: try basename in the manifest directory
                        candidate2 = manifest.parent / mp_raw.name
                        mp_resolved = candidate2 if candidate2.exists() else candidate
                layer_map[name] = mp_resolved
        else:
            # fallback: scan layer_*_metadata.json
            for p in sorted(path.glob("layer_*_metadata.json")):
                layer_map[p.stem.replace("_metadata", "")] = p
    else:
        # file: could be manifest or single layer metadata
        if path.name == "layers_manifest.json":
            data = _load_metadata(path)
            for l in data.get("layers", []):
                name = l.get("name")
                mp_raw = Path(l.get("metadata_path"))
                if mp_raw.is_absolute():
                    mp_resolved = mp_raw
                else:
                    candidate = path.parent / mp_raw
                    if candidate.exists():
                        mp_resolved = candidate
                    else:
                        # Fallback: try basename in the manifest directory
                        candidate2 = path.parent / mp_raw.name
                        mp_resolved = candidate2 if candidate2.exists() else candidate
                layer_map[name] = mp_resolved
        else:
            # single layer metadata
            name = path.stem.replace("_metadata", "")
            layer_map[name] = path

    options = sorted(layer_map.keys())
    return options, layer_map


def _draw_grid_on_image(img: Image.Image, H_ds: int, W_ds: int, line_color=(0, 255, 0), alpha: float = 0.5) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W_img, H_img = img.width, img.height
    cell_w = W_img / float(W_ds)
    cell_h = H_img / float(H_ds)

    # Draw vertical lines
    for c in range(1, W_ds):
        x = int(round(c * cell_w))
        draw.line([(x, 0), (x, H_img)], fill=(*line_color, int(255 * alpha)), width=1)

    # Draw horizontal lines
    for r in range(1, H_ds):
        y = int(round(r * cell_h))
        draw.line([(0, y), (W_img, y)], fill=(*line_color, int(255 * alpha)), width=1)

    result = Image.alpha_composite(img.convert("RGBA"), overlay)
    return result.convert("RGB")


def _overlay_rgba_heatmap(base: Image.Image, heatmap_rgba: Image.Image, opacity: float = 0.6) -> Image.Image:
    base_rgba = base.convert("RGBA")
    hm = heatmap_rgba.convert("RGBA")
    # Scale existing alpha by slider opacity
    r, g, b, a = hm.split()
    a_arr = (np.array(a).astype(np.float32) * float(np.clip(opacity, 0.0, 1.0)))
    a_scaled = Image.fromarray(np.clip(a_arr, 0, 255).astype(np.uint8), mode="L")
    hm_scaled = Image.merge("RGBA", (r, g, b, a_scaled))
    return Image.alpha_composite(base_rgba, hm_scaled).convert("RGB")


def _click_to_query(x: int, y: int, H_ds: int, W_ds: int, H_img: int, W_img: int) -> int:
    cell_w = W_img / float(W_ds)
    cell_h = H_img / float(H_ds)
    qc = int(np.clip(np.floor(x / cell_w), 0, W_ds - 1))
    qr = int(np.clip(np.floor(y / cell_h), 0, H_ds - 1))
    qi = int(qr * W_ds + qc)
    return qi


def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
        **VGGT Attention Explorer**  
        1) Provide the metadata JSON path produced by compute script.  
        2) Click a patch on the first image to view per-frame overlays.
        """)

        with gr.Row():
            metadata_tb = gr.Textbox(label="Attn maps directory or metadata path", placeholder="/path/to/attn_maps or layer_xx_metadata.json")
            load_btn = gr.Button("Load")

        opacity = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Overlay opacity")

        # Top controls row: Layer selector + Head selector + vmax display
        with gr.Row():
            with gr.Column(scale=1):
                layer_select = gr.Dropdown(label="Layer", choices=[], value=None, interactive=True)
            with gr.Column(scale=1):
                head_select = gr.Dropdown(label="Head", choices=["Average"], value="Average", interactive=True)
            with gr.Column(scale=3):
                vmax_html = gr.HTML(value="")

        # Content row: query image (left, large) | per-frame overlays (right, smaller)
        with gr.Row():
            with gr.Column(scale=3):
                img_click = gr.Image(type="pil", label="Click a query patch on the first image", interactive=True)
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Per-frame overlays", columns=5, rows=1, height=600)

        # Internal state
        state_meta = gr.State(value=None)  # current layer meta
        state_data = gr.State(value=None)  # {mode: 'npz'|'png', ...}
        state_npz = gr.State(value=None)   # loaded npz memmap (optional)
        state_layer_map = gr.State(value=None)  # name -> metadata path
        state_image_paths = gr.State(value=None)  # resolved absolute image paths

        def on_load(meta_or_dir_path: str):
            mp = Path(meta_or_dir_path.strip())
            if not mp.exists():
                raise gr.Error(f"Path not found: {mp}")
            options, layer_map = _discover_layers(mp)
            if len(options) == 0:
                raise gr.Error("No layers found in the provided path")
            # default to last layer
            selected = options[-1]
            meta = _load_metadata(layer_map[selected])
            image_paths, data = _resolve_paths(meta, layer_map[selected])
            if len(image_paths) == 0:
                raise gr.Error("No images found in metadata")

            first_img = Image.open(image_paths[0]).convert("RGB")
            first_img_grid = _draw_grid_on_image(first_img, meta["H_ds"], meta["W_ds"], line_color=(0, 255, 0), alpha=0.6)
            info = f"<div style=\"font-size: 48px; font-weight: 800;\">vmax (99%): {meta.get('layer_vmax99', 1.0):.4f}</div>"
            npz_loaded = None
            if data["mode"] == "npz":
                # Load lazily with mmap (works for both .npz and .npy)
                npz_loaded = np.load(data["path"], mmap_mode="r")
            
            # Setup head selector
            per_head = meta.get("per_head_data", False)
            if per_head:
                num_heads = meta.get("num_heads", 16)
                head_choices = ["Average"] + [f"Head {i+1}" for i in range(num_heads)]
            else:
                head_choices = ["Average"]
            
            return first_img_grid, meta, data, gr.update(value=None), info, gr.update(choices=options, value=selected), layer_map, npz_loaded, image_paths, gr.update(choices=head_choices, value="Average")

        load_btn.click(
            fn=on_load,
            inputs=[metadata_tb],
            outputs=[img_click, state_meta, state_data, gallery, vmax_html, layer_select, state_layer_map, state_npz, state_image_paths, head_select],
        )

        def on_change_layer(layer_name: str, layer_map: Dict[str, str]):
            if not layer_name or not layer_map:
                raise gr.Error("Load a directory/manifest first")
            meta_path = Path(layer_map[layer_name])
            meta = _load_metadata(meta_path)
            image_paths, data = _resolve_paths(meta, meta_path)
            first_img = Image.open(image_paths[0]).convert("RGB")
            first_img_grid = _draw_grid_on_image(first_img, meta["H_ds"], meta["W_ds"], line_color=(0, 255, 0), alpha=0.6)
            info = f"<div style=\"font-size: 48px; font-weight: 800;\">vmax (99%): {meta.get('layer_vmax99', 1.0):.4f}</div>"
            npz_loaded = None
            if data["mode"] == "npz":
                npz_loaded = np.load(data["path"], mmap_mode="r")
            
            # Setup head selector
            per_head = meta.get("per_head_data", False)
            if per_head:
                num_heads = meta.get("num_heads", 16)
                head_choices = ["Average"] + [f"Head {i+1}" for i in range(num_heads)]
            else:
                head_choices = ["Average"]
            
            return first_img_grid, meta, data, gr.update(value=None), info, npz_loaded, image_paths, gr.update(choices=head_choices, value="Average")

        layer_select.change(
            fn=on_change_layer,
            inputs=[layer_select, state_layer_map],
            outputs=[img_click, state_meta, state_data, gallery, vmax_html, state_npz, state_image_paths, head_select],
        )

        def on_click(img: Image.Image, meta: Dict[str, Any], data_state: Dict[str, Any], opacity_value: float, npz_loaded, image_paths, head_selected: str, evt: gr.SelectData):
            if meta is None or data_state is None:
                raise gr.Error("Load metadata first")
            if image_paths is None:
                raise gr.Error("Internal error: missing image paths")
            if len(image_paths) == 0:
                return []
            base0 = Image.open(image_paths[0]).convert("RGB")
            # evt.index is (x, y) in pixels for gr.Image.select
            x, y = evt.index
            qi = _click_to_query(x, y, meta["H_ds"], meta["W_ds"], base0.height, base0.width)

            # Parse head selection
            per_head = meta.get("per_head_data", False)
            if per_head and head_selected != "Average":
                # Extract head index from "Head X" format
                head_idx = int(head_selected.split()[-1]) - 1
            else:
                head_idx = None  # Use average

            overlays: List[Image.Image] = []
            if data_state["mode"] == "png":
                all_heatmaps = data_state["paths"]
                if qi < 0 or qi >= len(all_heatmaps):
                    return []
                for s, (base_path, hm_path) in enumerate(zip(image_paths, all_heatmaps[qi])):
                    base = Image.open(base_path).convert("RGB")
                    hm = Image.open(hm_path).convert("RGBA")
                    over = _overlay_rgba_heatmap(base, hm, opacity=opacity_value)
                    overlays.append(over)
                return overlays
            else:
                # npz path mode
                if npz_loaded is None:
                    npz_loaded = np.load(data_state["path"], mmap_mode="r")
                # Support .npz (dict-like) and .npy (ndarray)
                if hasattr(npz_loaded, "files"):
                    hm = npz_loaded["heatmaps"]  # uint8 [Q,S,Hq,Wq] or [Q,H,S,Hq,Wq]
                else:
                    hm = npz_loaded  # uint8 [Q,S,Hq,Wq] or [Q,H,S,Hq,Wq]
                
                # Handle per-head vs averaged data
                if per_head and hm.ndim == 5:  # [Q, num_heads, S, Hq, Wq]
                    if head_idx is not None:
                        # Select specific head
                        hm_query = hm[qi, head_idx]  # [S, Hq, Wq]
                    else:
                        # Average across heads
                        hm_query = hm[qi].mean(axis=0).astype(np.uint8)  # [S, Hq, Wq]
                else:  # [Q, S, Hq, Wq]
                    Q = hm.shape[0]
                    if qi < 0 or qi >= Q:
                        return []
                    hm_query = hm[qi]  # [S, Hq, Wq]
                
                # Build jet LUT
                lut = (cm.get_cmap("jet")(np.linspace(0, 1, 256))[:, :3] * 255.0).astype(np.uint8)
                for s, base_path in enumerate(image_paths):
                    base = Image.open(base_path).convert("RGB")
                    plane = hm_query[s]  # uint8 [Hq,Wq]
                    # Blockwise upsample (nearest neighbor) to image size
                    im_small = Image.fromarray(plane, mode="L")
                    im_full = im_small.resize((base.width, base.height), resample=Image.NEAREST)
                    plane_full = np.array(im_full, dtype=np.uint8)
                    rgb = lut[plane_full]
                    alpha = plane_full  # use normalized magnitude as alpha
                    rgba = np.dstack([rgb, alpha])
                    heat = Image.fromarray(rgba, mode="RGBA")
                    over = _overlay_rgba_heatmap(base, heat, opacity=opacity_value)
                    overlays.append(over)
                return overlays

        img_click.select(
            fn=on_click,
            inputs=[img_click, state_meta, state_data, opacity, state_npz, state_image_paths, head_select],
            outputs=[gallery],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=False, help="Path to layer_xx_metadata.json")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo = build_interface()
    if args.metadata:
        # Pre-fill textbox via queue hack: keep it simple; user can paste manually
        pass

    demo.queue().launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()


