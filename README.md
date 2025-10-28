# VGGT Attention Visualization Toolkit

Interactive, layer-wise visualization of VGGT global attention.

- Fast exporters that capture attention in a single forward pass
- Compact per-layer artifacts (low‑res heatmaps normalized by layer 99th percentile)
- A Gradio UI to click any query patch and see per‑frame overlays

<img width="1658" height="933" alt="Selection_652" src="https://github.com/user-attachments/assets/240002eb-1962-4093-becd-ebbd6a671c01" />

---


## Key scripts

- `vggt/utils/attn_capture.py` — hook utilities to collect attention during a single forward
- `compute_layerwise_attentions.py` — runs the model once and exports all layers
- `visualize_attn_gradio.py` — interactive viewer (click query patch → per‑frame overlays)

---

## Quick start

1) Export attention for all layers (also copies the images used):

```bash
python compute_layerwise_attentions.py \
  --image_set_name kitchen \
  --num_images 5 \
  --output_dir_prefix attn_first2all_ \
```

Notes:
- The script loads VGGT weights, picks `num_images` from `examples/<image_set_name>/images/`,
  and writes into `attn_first2all_<image_set_name>/`.
- Each layer gets a single compact NPZ with low‑res heatmaps normalized by that layer’s 99th percentile (`layer_vmax99`).

2) Launch the visualizer:

```bash
python visualize_attn_gradio.py --server_port 7860
```

Then paste the `attn_maps/` directory (or `layers_manifest.json`) into the textbox and click “Load”.

- Choose a layer from the dropdown (top‑left).
- Click a query patch on the large first image (left).
- The S per‑frame overlays appear on the right, jet‑colored with per‑pixel alpha.

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

- `heatmaps_uint8.npz` — `heatmaps` shaped `[Q, S, Hq, Wq]` (uint8)
  - Q: number of query patches in frame 0 (row‑major over the downsampled grid)
  - S: number of frames/images
  - Hq×Wq: query grid resolution after stride (`stride_h`, `stride_w`)
- `layer_XX_metadata.json` — pretty‑printed JSON with:
  - `npz_path`: relative path to the NPZ (relative to `attn_maps/`)
  - `S`, `H_ds`, `W_ds`, `stride_h`, `stride_w`
  - `layer_vmax99`: layerwise 99th percentile used for normalization
  - `image_paths`: paths to copied images (relative to `attn_maps/`)

`layers_manifest.json` lists all exported layers and their metadata paths.

---

## License

This project follows the license of the underlying VGGT repository and model weights. Review those terms before redistribution.
