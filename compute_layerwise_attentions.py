import argparse
import os
from pathlib import Path
import shutil
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import vggt.utils.attn_capture as attn_capture
from vggt.utils.attn_capture import capture_and_save_first_to_all_attention

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

def compute_layerwise_attentions(
    model, 
    images, 
    image_set_name="kitchen", 
    output_dir="attn_outputs_first2all"
):
    saved_first2all = capture_and_save_first_to_all_attention(
        model,
        images,
        image_set_name=image_set_name,
        output_dir=output_dir,
        stride_h=1,
        stride_w=1,
        amp_dtype=dtype,
        max_query_chunk=2048,
    )
    return saved_first2all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_set_name", type=str, default="kitchen")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="attn_outputs_first2all")
    args = parser.parse_args()
    
    model = load_model()
    
    image_dir = Path(f"examples/{args.image_set_name}/images")
    all_image_names = [image_dir / f for f in image_dir.glob("*.png")]
    step = len(all_image_names) // args.num_images
    # take every step-th image
    image_names = [str(all_image_names[i]) for i in range(0, len(all_image_names), step)]
    images = load_images(image_names)
    
    # copy images to output_dir / images
    os.makedirs(Path(args.output_dir) / "images", exist_ok=True)
    for image_name in image_names:
        shutil.copy(image_name, Path(args.output_dir) / "images" / os.path.basename(image_name))
    
    saved_first2all = compute_layerwise_attentions(model, images, args.image_set_name, args.output_dir)
    print(saved_first2all)
    