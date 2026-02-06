"""
In-memory (RAM) inference demo for DiffuEraser.

Reads example2 video+mask into memory once, runs:
  1) ProPainter -> priori_frames (in memory)
  2) DiffuEraser -> output frames (in memory)

It does NOT write any output video by default. This is intended to be used with an
async/queued writer on the caller side to avoid slow disk I/O stalling the GPU.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import torch
import torchvision
from PIL import Image

from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device


def decode_video_to_pil_frames(
    video_path: str,
    video_length_sec: float,
) -> Tuple[List[Image.Image], float]:
    vframes, _, info = torchvision.io.read_video(
        filename=video_path, pts_unit="sec", end_pts=video_length_sec
    )  # RGB uint8 tensor [T,H,W,C]
    fps = float(info["video_fps"])
    n_total_frames = int(video_length_sec * fps)
    frames = list(vframes.numpy())[:n_total_frames]
    frames = [Image.fromarray(f) for f in frames]
    return frames, fps


def run_inmem_inference(
    *,
    input_video: str,
    input_mask: str,
    video_length: float,
    max_img_size: int = 960,
    mask_dilation_iter: int = 8,
    ref_stride: int = 10,
    neighbor_length: int = 10,
    subvideo_length: int = 50,
    base_model_path: str = "weights/stable-diffusion-v1-5",
    vae_path: str = "weights/sd-vae-ft-mse",
    diffueraser_path: str = "weights/diffuEraser",
    propainter_model_dir: str = "weights/propainter",
    seed: Optional[int] = None,
    blended: bool = True,
    empty_cache: bool = False,
    profile: bool = False,
    profile_sync: bool = False,
    profile_log_path: Optional[str] = None,
) -> List[Image.Image]:
    """
    Returns:
        List[PIL.Image.Image]: output frames (RGB) after composition.
    """
    device = get_device()

    # Pre-decode video+mask once into RAM.
    frames, fps = decode_video_to_pil_frames(input_video, video_length)
    mask_frames, mask_fps = decode_video_to_pil_frames(input_mask, video_length)
    if abs(mask_fps - fps) > 1e-3:
        raise ValueError(f"Mask fps ({mask_fps}) != video fps ({fps}). Please ensure they match.")
    mask_frames = mask_frames[: len(frames)]

    # Init models once.
    ckpt = "2-Step"
    video_inpainting_sd = DiffuEraser(device, base_model_path, vae_path, diffueraser_path, ckpt=ckpt)
    propainter = Propainter(propainter_model_dir, device=device)

    # 1) ProPainter -> priori frames (RAM)
    priori_frames = propainter.forward(
        input_video,
        mask_frames,
        output_path=None,
        input_frames=frames,
        input_fps=fps,
        video_length=video_length,
        ref_stride=ref_stride,
        neighbor_length=neighbor_length,
        subvideo_length=subvideo_length,
        mask_dilation=mask_dilation_iter,
        save_video=False,
        return_priori_frames=True,
        empty_cache=empty_cache,
        profile=profile,
        profile_sync=profile_sync,
        profile_log_path=profile_log_path,
    )

    # 2) DiffuEraser -> output frames (RAM)
    out_frames = video_inpainting_sd.forward(
        frames,
        mask_frames,
        priori_frames,
        output_path=None,
        max_img_size=max_img_size,
        video_length=video_length,
        mask_dilation_iter=mask_dilation_iter,
        seed=seed,
        input_fps=fps,
        empty_cache=empty_cache,
        save_video=False,
        return_frames=True,
        profile=profile,
        profile_sync=profile_sync,
        profile_log_path=profile_log_path,
        blended=blended,
    )
    assert isinstance(out_frames, list)
    return out_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, default="examples/example2/video.mp4")
    parser.add_argument("--input_mask", type=str, default="examples/example2/mask.mp4")
    parser.add_argument("--video_length", type=float, default=2)
    parser.add_argument("--max_img_size", type=int, default=960)
    parser.add_argument("--mask_dilation_iter", type=int, default=8)
    parser.add_argument("--no_blend", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable_empty_cache", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_sync", action="store_true")
    parser.add_argument("--profile_log_path", type=str, default=None)
    args = parser.parse_args()

    # Help catch accidental CPU-only runs; this script is intended for GPU throughput testing.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU node / with correct env.")

    frames = run_inmem_inference(
        input_video=args.input_video,
        input_mask=args.input_mask,
        video_length=args.video_length,
        max_img_size=args.max_img_size,
        mask_dilation_iter=args.mask_dilation_iter,
        seed=args.seed,
        blended=not args.no_blend,
        empty_cache=not args.disable_empty_cache,
        profile=args.profile,
        profile_sync=args.profile_sync,
        profile_log_path=args.profile_log_path,
    )
    print(f"[inmem] done: out_frames={len(frames)} size={frames[0].size if frames else None}")


if __name__ == "__main__":
    main()


