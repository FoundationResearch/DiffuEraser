import torch
import os 
import time
import argparse
import torchvision
from PIL import Image
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device

def decode_video_once(video_path: str, video_length_sec: float):
    """
    Decode video once and return PIL RGB frames + fps.
    """
    vframes, _, info = torchvision.io.read_video(
        filename=video_path, pts_unit="sec", end_pts=video_length_sec
    )  # RGB, uint8 tensor [T,H,W,C]
    fps = info["video_fps"]
    n_total_frames = int(video_length_sec * fps)
    frames = list(vframes.numpy())[:n_total_frames]
    frames = [Image.fromarray(f) for f in frames]
    return frames, fps

def main():

    ## input params
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default="examples/example3/video.mp4", help='Path to the input video')
    parser.add_argument('--input_mask', type=str, default="examples/example3/mask.mp4" , help='Path to the input mask')
    parser.add_argument('--video_length', type=int, default=10, help='The maximum length of output video')
    parser.add_argument('--mask_dilation_iter', type=int, default=8, help='Adjust it to change the degree of mask expansion')
    parser.add_argument('--max_img_size', type=int, default=960, help='The maximum length of output width and height')
    parser.add_argument('--save_path', type=str, default="results" , help='Path to the output')
    parser.add_argument('--output_name', type=str, default="diffueraser_result.mp4", help='Output video filename under save_path')
    parser.add_argument('--save_priori_video', action='store_true', help='Also save ProPainter output video (priori.mp4) under save_path for debugging.')
    parser.add_argument('--no_blend', action='store_true', help='Disable blurred mask blending in DiffuEraser compose stage (faster CPU, slightly harsher boundaries).')
    parser.add_argument('--ref_stride', type=int, default=10, help='Propainter params')
    parser.add_argument('--neighbor_length', type=int, default=10, help='Propainter params')
    parser.add_argument('--subvideo_length', type=int, default=50, help='Propainter params')
    parser.add_argument('--base_model_path', type=str, default="weights/stable-diffusion-v1-5" , help='Path to sd1.5 base model')
    parser.add_argument('--vae_path', type=str, default="weights/sd-vae-ft-mse" , help='Path to vae')
    parser.add_argument('--diffueraser_path', type=str, default="weights/diffuEraser" , help='Path to DiffuEraser')
    parser.add_argument('--propainter_model_dir', type=str, default="weights/propainter" , help='Path to priori model')
    parser.add_argument('--disable_empty_cache', action='store_true', help='Disable frequent torch.cuda.empty_cache/gc.collect calls for better throughput (may increase peak VRAM).')
    parser.add_argument('--profile', action='store_true', help='Print stage timing to help diagnose CPU/GPU stalls.')
    parser.add_argument('--profile_sync', action='store_true', help='If set, synchronize CUDA before timing points for more accurate GPU step timings (adds overhead).')
    args = parser.parse_args()
                  
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    priori_path = os.path.join(args.save_path, "priori.mp4")
    output_path = os.path.join(args.save_path, args.output_name)
    profile_log_path = os.path.join(args.save_path, "profile.log")
    print(f"[run] save_path={os.path.abspath(args.save_path)}")
    print(f"[run] output_path={os.path.abspath(output_path)}")
    if args.save_priori_video:
        print(f"[run] priori_path={os.path.abspath(priori_path)}")
    if args.profile:
        print(f"[run] profile_log={os.path.abspath(profile_log_path)}")
    
    ## model initialization
    init_t0 = time.time()
    device = get_device()
    # PCM params
    ckpt = "2-Step"
    video_inpainting_sd = DiffuEraser(device, args.base_model_path, args.vae_path, args.diffueraser_path, ckpt=ckpt)
    propainter = Propainter(args.propainter_model_dir, device=device)
    init_t1 = time.time()
    print(f"[run] model init time: {init_t1 - init_t0:.4f} s")
    
    start_time = time.time()
    def sync():
        if args.profile_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def log_step(msg: str):
        if args.profile:
            sync()
            dt = time.time() - start_time
            line = f"[run][{dt:8.3f}s] {msg}"
            print(line)
            with open(profile_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # Decode input video only once (both Propainter and DiffuEraser need frames).
    # Fallback to old path if input is not a video file path.
    shared_frames, shared_fps = None, None
    if args.input_video.endswith(("mp4", "mov", "avi", "MP4", "MOV", "AVI")):
        log_step("decode_video_once: start")
        shared_frames, shared_fps = decode_video_once(args.input_video, args.video_length)
        log_step(f"decode_video_once: done (frames={len(shared_frames)}, fps={shared_fps})")

    ## priori (in-memory) - avoid writing/reading intermediate `priori.mp4`
    log_step("Propainter.forward: start")
    priori_frames = propainter.forward(
        args.input_video,
        args.input_mask,
        output_path=priori_path if args.save_priori_video else None,
        input_frames=shared_frames,
        input_fps=shared_fps,
        video_length=args.video_length,
        ref_stride=args.ref_stride,
        neighbor_length=args.neighbor_length,
        subvideo_length=args.subvideo_length,
        mask_dilation=args.mask_dilation_iter,
        save_video=args.save_priori_video,
        return_priori_frames=True,
        empty_cache=not args.disable_empty_cache,
        profile=args.profile,
        profile_sync=args.profile_sync,
        profile_log_path=profile_log_path,
    )
    log_step(f"Propainter.forward: done (priori_frames={len(priori_frames)})")

    ## diffueraser
    guidance_scale = None    # The default value is 0.  
    log_step("DiffuEraser.forward: start")
    video_inpainting_sd.forward(
        shared_frames if shared_frames is not None else args.input_video,
        args.input_mask,
        priori_frames,
        output_path,
                                max_img_size = args.max_img_size, video_length=args.video_length, mask_dilation_iter=args.mask_dilation_iter,
                                guidance_scale=guidance_scale,
                                input_fps=shared_fps,
                                empty_cache=not args.disable_empty_cache,
                                profile=args.profile,
                                profile_sync=args.profile_sync,
                                profile_log_path=profile_log_path,
                                blended=not args.no_blend,
    )
    log_step("DiffuEraser.forward: done")
    
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"[run] end-to-end (excluding model init) time: {inference_time:.4f} s")
    if args.profile:
        with open(profile_log_path, "a", encoding="utf-8") as f:
            f.write(f"model_init_sec={init_t1-init_t0:.6f}\n")
            f.write(f"e2e_excl_init_sec={inference_time:.6f}\n")
            f.write(f"save_path={os.path.abspath(args.save_path)}\n")
            f.write(f"output_path={os.path.abspath(output_path)}\n")
            if args.save_priori_video:
                f.write(f"priori_path={os.path.abspath(priori_path)}\n")
            f.write(f"disable_empty_cache={bool(args.disable_empty_cache)}\n")
            f.write(f"no_blend={bool(args.no_blend)}\n")
            f.write("----\n")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


   