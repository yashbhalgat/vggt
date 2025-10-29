# VGGT Attention Visualization Toolkit

Interactive, layer-wise visualization of VGGT global attention with per-head analysis.

- Fast exporters that capture attention in a single forward pass
- **Per-head attention support** with optimal quantization and dynamic visualization
- Compact per-layer artifacts (low‑res heatmaps with per-head or layer normalization)
- A Gradio UI to click any query patch and see per‑frame overlays
- **Compare all layers or all heads simultaneously** with organized composite views

<img width="1658" height="933" alt="Selection_652" src="https://github.com/user-attachments/assets/240002eb-1962-4093-becd-ebbd6a671c01" />

---

## Key scripts

- `vggt/utils/attn_capture.py` - hook utilities to collect attention during a single forward
- `compute_layerwise_attentions.py` - runs the model once and exports all layers
- `visualize_attn_gradio.py` - interactive viewer (click query patch → per‑frame overlays)

---

## Quick start

1) Export attention for all layers (also copies the images used):

```bash
# Basic usage (averaged attention)
python compute_layerwise_attentions.py \
  --image_set_name kitchen \
  --num_images 5 \
  --output_dir_prefix attn_first2all_

# With per-head attention (new feature)
python compute_layerwise_attentions.py \
  --image_set_name kitchen \
  --num_images 5 \
  --save_per_head \
  --output_dir_prefix attn_first2all_

# Using new parameters
python compute_layerwise_attentions.py \
  --image_set_name room \
  --image_dir examples/room/images \
  --image_extension jpg \
  --first_image_idx 2 \
  --save_per_head \
  --output_dir_prefix attn_first2all_
```

**New Parameters:**
- `--save_per_head`: Capture all 16 attention heads separately (each normalized by its own vmax99)
- `--image_extension`: Specify image format to load (png/jpg/jpeg, default: png)
- `--first_image_idx`: Choose which image to use as query frame (default: 0)

**Notes:**
- The script loads VGGT weights, picks `num_images` from `examples/<image_set_name>/images/`,
  and writes into `attn_first2all_<image_set_name>/`.
- With `--save_per_head`, each head is normalized by its own vmax99 for optimal quantization.
- Without `--save_per_head`, heatmaps are normalized by layer-wide vmax99.

2) Launch the visualizer:

```bash
python visualize_attn_gradio.py --server_port 7860
```

Then paste the `attn_maps/` directory (or `layers_manifest.json`) into the textbox and click "Load".

- Choose a layer from the dropdown (top‑left).
- **New:** Choose a head from the dropdown (Average, or Head 1-16 if per-head data available).
- Click a query patch on the large first image (left).
- The S per‑frame overlays appear on the right, jet‑colored with per‑pixel alpha.
- **New:** Check "Show all layers at once" to view all layers in vertical composites.
- **New:** Check "Show all heads at once" to view all heads in vertical composites.

**New UI Elements:**
- **Head selector dropdown**: Switch between "Average" and individual heads (Head 1-16) when per-head data available
- **Show all layers at once checkbox**: View all layers simultaneously in organized vertical composites
- **Show all heads at once checkbox**: View all heads simultaneously in organized vertical composites
- **Dynamic vmax display**: Shows per-head statistics that update based on selection

**Per-Head Visualization Behavior:**
- Individual heads: displayed with optimal per-head normalization
- Average view: heads averaged in raw scale, then renormalized for display
- Each head's vmax99 shown in composite titles

---

## Output layout

Exported attention maps live under an output root, for example `attn_first2all_kitchen/`:

```
attn_first2all_kitchen/
  images/                        # copies of the input images used
  attn_maps/
    layers_manifest.json         # list of layers and metadata paths
    layer_01/
      heatmaps_uint8.npz         # uint8 [Q, S, Hq, Wq]
    layer_01_metadata.json       # metadata (paths, Hq/Wq, vmax, etc.)
    layer_02/
      heatmaps_uint8.npz
    layer_02_metadata.json
    ...
```

Per layer (`layer_XX/`):

- `heatmaps_uint8.npz` — `heatmaps` shaped `[Q, S, Hq, Wq]` or `[Q, num_heads, S, Hq, Wq]` (uint8)
  - Q: number of query patches in frame 0 (row‑major over the downsampled grid)
  - num_heads: 16 attention heads (only if `--save_per_head` was used)
  - S: number of frames/images
  - Hq×Wq: query grid resolution after stride (`stride_h`, `stride_w`)
- `layer_XX_metadata.json` — pretty‑printed JSON with:
  - `npz_path`: relative path to the NPZ (relative to `attn_maps/`)
  - `S`, `H_ds`, `W_ds`, `stride_h`, `stride_w`
  - `layer_vmax99`: layerwise 99th percentile (across all heads if per-head)
  - `per_head_data`: boolean indicating if per-head data is available
  - `num_heads`: number of heads (if per-head data)
  - `per_head_vmax99`: array of per-head 99th percentiles (if per-head data)
  - `normalized_by`: "per_head_vmax99" or "layer_vmax99"
  - `image_paths`: paths to copied images (relative to `attn_maps/`)

`layers_manifest.json` lists all exported layers and their metadata paths.

---

## Changelog

### 29/10/2025
- Added `--save_per_head` flag to capture all 16 attention heads separately with per-head normalization
- Added head selector dropdown in UI to switch between "Average" and individual heads
- Added `--first_image_idx` parameter to choose query frame
- Added `--image_extension` parameter for png/jpg/jpeg support
- Added "Show all layers at once" and "Show all heads at once" checkboxes for composite views
- Per-head data uses optimal quantization (each head normalized by its own vmax99)
- Storage format extended to `[Q, num_heads, S, H_ds, W_ds]` with per_head_vmax99 metadata

---

## License

This project follows the license of the underlying VGGT repository and model weights. Review those terms before redistribution.
