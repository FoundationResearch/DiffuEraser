# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
import gc
import time

try:
    from model.modules.flow_comp_raft import RAFT_bi
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from model.propainter import InpaintGenerator
    from utils.download_util import load_file_from_url
    from model.misc import get_device
except Exception:
    from propainter.model.modules.flow_comp_raft import RAFT_bi
    from propainter.model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from propainter.model.propainter import InpaintGenerator
    from propainter.utils.download_util import load_file_from_url
    from propainter.model.misc import get_device

import warnings
warnings.filterwarnings("ignore")

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
MaxSideThresh = 960


# resize frames
def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size

#  read frames from video
def read_frame_from_videos(frame_root, video_length):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec', end_pts=video_length) # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
        nframes = len(frames)
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
        nframes = len(frames)
    size = frames[0].size

    return frames, fps, size, video_name, nframes

def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
# read frame-wise masks
def _dilate_binary_mask_opencv(mask_u8: np.ndarray, iterations: int) -> np.ndarray:
    """
    Fast binary dilation on CPU using OpenCV.
    We use a cross-shaped 3x3 kernel to better match scipy.ndimage.binary_dilation default connectivity=1.
    Returns uint8 array with values {0,1}.
    """
    if iterations <= 0:
        return (mask_u8 > 0).astype(np.uint8)
    m = (mask_u8 > 0).astype(np.uint8)
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    m = cv2.dilate(m, kernel, iterations=int(iterations))
    return m.astype(np.uint8)


def read_mask(mpath, frames_len, size, flow_mask_dilates=8, mask_dilates=5, use_opencv: bool = True):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    # In-memory masks: list/tuple of PIL/np arrays (already decoded elsewhere)
    if isinstance(mpath, (list, tuple)):
        masks_img = list(mpath)[:frames_len]
        masks_img = [m if isinstance(m, Image.Image) else Image.fromarray(np.asarray(m)) for m in masks_img]
    elif mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):  # input single img path
        masks_img = [Image.open(mpath)]
    elif mpath.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):  # input video path
        cap = cv2.VideoCapture(mpath)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx >= frames_len:
                break
            masks_img.append(Image.fromarray(frame))
            idx += 1
        cap.release()
    else:
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
          
    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            if use_opencv:
                flow_mask_img = _dilate_binary_mask_opencv(mask_img, iterations=flow_mask_dilates)
            else:
                flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            if use_opencv:
                mask_img = _dilate_binary_mask_opencv(mask_img, iterations=mask_dilates)
            else:
                mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * frames_len
        masks_dilated = masks_dilated * frames_len

    return flow_masks, masks_dilated

def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


class Propainter:
    def __init__(
            self, propainter_model_dir, device):
        # Normalize device input to torch.device for robustness (callers may pass "cuda"/"cuda:0"/"cpu" strings).
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        ##############################################
        # set up RAFT and flow competition model
        ##############################################
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                        model_dir=propainter_model_dir, progress=True, file_name=None)
        self.fix_raft = RAFT_bi(ckpt_path, device)
        
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                        model_dir=propainter_model_dir, progress=True, file_name=None)
        self.fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(device)
        self.fix_flow_complete.eval()

        ##############################################
        # set up ProPainter model
        ##############################################
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                        model_dir=propainter_model_dir, progress=True, file_name=None)
        self.model = InpaintGenerator(model_path=ckpt_path).to(device)
        self.model.eval()
    def forward(self, video, mask, output_path, resize_ratio=1.0, video_length=2, height=-1, width=-1,
                mask_dilation=4, ref_stride=10, neighbor_length=10, subvideo_length=80,
                raft_iter=20, save_fps=24, save_frames=False, fp16=True,
                save_video: bool = True,
                return_priori_frames: bool = False,
                input_frames=None,
                input_fps=None,
                empty_cache: bool = True,
                profile: bool = False,
                profile_sync: bool = False,
                profile_log_path: str = None,
                pin_memory: bool = True,
                non_blocking: bool = True,
                use_opencv_mask_dilate: bool = True,
                max_side_thresh: int = MaxSideThresh,
                raft_clip_len: int = 0,
                enable_preprop: bool = False):
        
        t0 = time.perf_counter()
        def log(msg: str):
            if profile:
                if profile_sync and self.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                line = f"[ProPainter][{dt:8.3f}s] {msg}"
                print(line)
                if profile_log_path:
                    with open(profile_log_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")

        def maybe_empty_cache():
            if empty_cache and self.device.type == "cuda":
                torch.cuda.empty_cache()

        def maybe_gc():
            if empty_cache:
                gc.collect()

        log(f"device={self.device}, fp16={fp16}")
        if self.device.type != "cuda":
            raise RuntimeError("This optimized ProPainter path requires a CUDA device.")
        # Performance-only: forbid writing priors to disk.
        if save_video or output_path is not None:
            raise RuntimeError("Saving ProPainter priors to disk is disabled. Do not pass `--save_priori_video`.")

        # Use fp16 precision during inference to reduce running memory cost
        use_half = True if fp16 else False 
        if self.device.type == 'cpu':
            use_half = False

        ################ read input video ################
        if input_frames is not None:
            # Use shared decoded frames (avoid decoding the same video twice).
            frames = list(input_frames)
            fps = input_fps
            if fps is None:
                fps = save_fps
            if video_length is not None and fps is not None:
                n_total_frames = int(video_length * fps)
                frames = frames[:n_total_frames]
            size = frames[0].size
            video_name = "in_memory"
            nframes = len(frames)
        else:
            frames, fps, size, video_name, nframes = read_frame_from_videos(video, video_length)
            frames = frames[:nframes]
        log(f"input frames={len(frames)}, fps={fps}, size={size}")
        if not width == -1 and not height == -1:
            size = (width, height)

        orig_size = size
        longer_edge = max(size[0], size[1])
        if(longer_edge > max_side_thresh): 
            scale = max_side_thresh / longer_edge
            resize_ratio = resize_ratio * scale
        if not resize_ratio == 1.0:
            size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))

        frames, size, out_size = resize_frames(frames, size)
        if size != orig_size:
            print(
                f"[ProPainter] resize input: orig={orig_size} -> process={size} "
                f"(out_size={out_size}, resize_ratio={resize_ratio:.4f}, MaxSideThresh={max_side_thresh})"
            )
        log(f"after resize: process_size={size}, out_size={out_size}")
        fps = save_fps if fps is None else fps

        ################ read mask ################ 
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(mask, frames_len, size,
                                            flow_mask_dilates=mask_dilation,
                                            mask_dilates=mask_dilation,
                                            use_opencv=use_opencv_mask_dilate)
        flow_masks = flow_masks[:nframes]
        masks_dilated = masks_dilated[:nframes]
        w, h = size
        log("mask read + dilation done")

        ################ adjust input ################ 
        frames_len = min(len(frames), len(masks_dilated))
        frames = frames[:frames_len]
        flow_masks = flow_masks[:frames_len]
        masks_dilated = masks_dilated[:frames_len]
        
        # ---------- fast CPU->GPU handoff ----------
        # The original `to_tensors()` path does per-frame PIL->bytes->transpose and is very slow for long videos.
        # Here we stack uint8 on CPU, optionally pin it, then transfer once and normalize/cast on GPU.
        log("prepare tensors (cpu): start")
        ori_frames_inp = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)  # [T,H,W,3] uint8
        flow_masks_u8 = np.stack([np.asarray(m.convert("L"), dtype=np.uint8) for m in flow_masks], axis=0)  # [T,H,W]
        masks_dilated_u8 = np.stack([np.asarray(m.convert("L"), dtype=np.uint8) for m in masks_dilated], axis=0)  # [T,H,W]

        frames_u8 = torch.from_numpy(ori_frames_inp).permute(0, 3, 1, 2).contiguous().unsqueeze(0)  # [1,T,3,H,W] u8
        flow_masks_u8 = torch.from_numpy(flow_masks_u8).unsqueeze(1).contiguous().unsqueeze(0)  # [1,T,1,H,W] u8
        masks_dilated_u8 = torch.from_numpy(masks_dilated_u8).unsqueeze(1).contiguous().unsqueeze(0)  # [1,T,1,H,W] u8
        log("prepare tensors (cpu): done")

        # Pin memory enables faster async H2D copies (only meaningful for CUDA).
        if self.device.type == "cuda" and pin_memory:
            frames_u8 = frames_u8.pin_memory()
            flow_masks_u8 = flow_masks_u8.pin_memory()
            masks_dilated_u8 = masks_dilated_u8.pin_memory()

        log("H2D copy: start")
        frames_u8 = frames_u8.to(self.device, non_blocking=non_blocking)
        flow_masks_u8 = flow_masks_u8.to(self.device, non_blocking=non_blocking)
        masks_dilated_u8 = masks_dilated_u8.to(self.device, non_blocking=non_blocking)
        log("H2D copy: done")

        log("normalize/cast on device: start")
        # frames in [-1, 1] float32 for RAFT; masks in [0, 1] float32
        frames = frames_u8.to(dtype=torch.float32).div_(127.5).sub_(1.0)
        flow_masks = flow_masks_u8.to(dtype=torch.float32).div_(255.0)
        masks_dilated = masks_dilated_u8.to(dtype=torch.float32).div_(255.0)
        log("normalize/cast on device: done")

        # Keep original uint8 frames on GPU for fast compose (avoid `.cpu()` inside the transformer loop).
        ori_frames_u8 = frames_u8  # [1,T,3,H,W] uint8 on device
        del flow_masks_u8, masks_dilated_u8
        log("moved tensors to device")
 
        ##############################################
        # ProPainter inference
        ##############################################
        video_length = frames.size(1)
        print(f'Priori generating: [{video_length} frames]...')
        with torch.no_grad():
            # ---- compute flow ----
            new_longer_edge = max(frames.size(-1), frames.size(-2))
            if raft_clip_len and int(raft_clip_len) > 0:
                short_clip_len = int(raft_clip_len)
            else:
                if new_longer_edge <= 640: 
                    short_clip_len = 12
                elif new_longer_edge <= 720: 
                    short_clip_len = 8
                elif new_longer_edge <= 1280:
                    short_clip_len = 4
                else:
                    short_clip_len = 2

            # use fp32 for RAFT
            if frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.fix_raft(frames[:,f:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = self.fix_raft(frames[:,f-1:end_f], iters=raft_iter)
                    
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    maybe_empty_cache()
                    
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(frames, iters=raft_iter)
                maybe_empty_cache()
            maybe_empty_cache()
            maybe_gc()
            log("RAFT flow computed")

            if use_half:
                frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self.fix_flow_complete = self.fix_flow_complete.half()
                self.model = self.model.half()
          
            # ---- complete flow ----
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + subvideo_length)
                    pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                        flow_masks[:, s_f:e_f+1])
                    pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                        pred_flows_bi_sub, 
                        flow_masks[:, s_f:e_f+1])

                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                    maybe_empty_cache()
                    
                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                maybe_empty_cache()
            maybe_empty_cache()
            maybe_gc()
            log("flow completion done")
                

            masks_dilated_ori = masks_dilated.clone()
            # ---- Pre-propagation (optional; expensive on long videos) ----
            subvideo_length_img_prop = min(100, subvideo_length) # ensure a minimum of 100 frames for image propagation
            if enable_preprop and (len(frames[0]) > subvideo_length_img_prop):
                sample_rate = len(frames[0])//(subvideo_length_img_prop//2)
                index_sample =  list(range(0, len(frames[0]), sample_rate))
                sample_frames =  torch.stack([frames[0][i].to(torch.float32) for i in index_sample]).unsqueeze(0) # use fp32 for RAFT
                sample_masks_dilated = torch.stack([masks_dilated[0][i] for i in index_sample]).unsqueeze(0)
                sample_flow_masks =  torch.stack([flow_masks[0][i] for i in index_sample]).unsqueeze(0)
  
                ## recompute flow for sampled frames
                # use fp32 for RAFT
                sample_video_length = sample_frames.size(1)
                if sample_frames.size(1) > short_clip_len:
                    gt_flows_f_list, gt_flows_b_list = [], []
                    for f in range(0, sample_video_length, short_clip_len):
                        end_f = min(sample_video_length, f + short_clip_len)
                        if f == 0:
                            flows_f, flows_b = self.fix_raft(sample_frames[:,f:end_f], iters=raft_iter)
                        else:
                            flows_f, flows_b = self.fix_raft(sample_frames[:,f-1:end_f], iters=raft_iter)
                        
                        gt_flows_f_list.append(flows_f)
                        gt_flows_b_list.append(flows_b)
                        maybe_empty_cache()
                        
                    gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                    gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                    sample_gt_flows_bi = (gt_flows_f, gt_flows_b)
                else:
                    sample_gt_flows_bi = self.fix_raft(sample_frames, iters=raft_iter)
                    maybe_empty_cache()
                maybe_empty_cache()
                maybe_gc()
                log("sample RAFT flow computed (pre-prop)")

                if use_half:
                    sample_frames, sample_flow_masks, sample_masks_dilated = sample_frames.half(), sample_flow_masks.half(), sample_masks_dilated.half()
                    sample_gt_flows_bi = (sample_gt_flows_bi[0].half(), sample_gt_flows_bi[1].half())

                # ---- complete flow ----
                flow_length = sample_gt_flows_bi[0].size(1)
                if flow_length > subvideo_length:
                    pred_flows_f, pred_flows_b = [], []
                    pad_len = 5
                    for f in range(0, flow_length, subvideo_length):
                        s_f = max(0, f - pad_len)
                        e_f = min(flow_length, f + subvideo_length + pad_len)
                        pad_len_s = max(0, f) - s_f
                        pad_len_e = e_f - min(flow_length, f + subvideo_length)
                        pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                            (sample_gt_flows_bi[0][:, s_f:e_f], sample_gt_flows_bi[1][:, s_f:e_f]), 
                            sample_flow_masks[:, s_f:e_f+1])
                        pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                            (sample_gt_flows_bi[0][:, s_f:e_f], sample_gt_flows_bi[1][:, s_f:e_f]), 
                            pred_flows_bi_sub, 
                            sample_flow_masks[:, s_f:e_f+1])

                        pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                        pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                        maybe_empty_cache()
                        
                    pred_flows_f = torch.cat(pred_flows_f, dim=1)
                    pred_flows_b = torch.cat(pred_flows_b, dim=1)
                    sample_pred_flows_bi = (pred_flows_f, pred_flows_b)
                else:
                    sample_pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(sample_gt_flows_bi, sample_flow_masks)
                    sample_pred_flows_bi = self.fix_flow_complete.combine_flow(sample_gt_flows_bi, sample_pred_flows_bi, sample_flow_masks)
                    maybe_empty_cache()
                maybe_empty_cache()
                maybe_gc()
                log("sample flow completion done (pre-prop)")
                
                masked_frames = sample_frames * (1 - sample_masks_dilated)
                
                if sample_video_length > subvideo_length_img_prop:
                    updated_frames, updated_masks = [], []
                    pad_len = 10
                    for f in range(0, sample_video_length, subvideo_length_img_prop):
                        s_f = max(0, f - pad_len)
                        e_f = min(sample_video_length, f + subvideo_length_img_prop + pad_len)
                        pad_len_s = max(0, f) - s_f
                        pad_len_e = e_f - min(sample_video_length, f + subvideo_length_img_prop)

                        b, t, _, _, _ = sample_masks_dilated[:, s_f:e_f].size()
                        pred_flows_bi_sub = (sample_pred_flows_bi[0][:, s_f:e_f-1], sample_pred_flows_bi[1][:, s_f:e_f-1])
                        prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                            pred_flows_bi_sub, 
                                                                            sample_masks_dilated[:, s_f:e_f], 
                                                                            'nearest')
                        updated_frames_sub = sample_frames[:, s_f:e_f] * (1 - sample_masks_dilated[:, s_f:e_f]) + \
                                            prop_imgs_sub.view(b, t, 3, h, w) * sample_masks_dilated[:, s_f:e_f]
                        updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                        
                        updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                        updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                        maybe_empty_cache()
                        
                    updated_frames = torch.cat(updated_frames, dim=1)
                    updated_masks = torch.cat(updated_masks, dim=1)
                else:
                    b, t, _, _, _ = sample_masks_dilated.size()
                    prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, sample_pred_flows_bi, sample_masks_dilated, 'nearest')
                    updated_frames = sample_frames * (1 - sample_masks_dilated) + prop_imgs.view(b, t, 3, h, w) * sample_masks_dilated
                    updated_masks = updated_local_masks.view(b, t, 1, h, w)
                    maybe_empty_cache()

                ## replace input frames/masks with updated frames/masks 
                for i,index in enumerate(index_sample):
                    frames[0][index] = updated_frames[0][i]
                    masks_dilated[0][index] = updated_masks[0][i]


            # ---- frame-by-frame image propagation ----
            masked_frames = frames * (1 - masks_dilated)
            subvideo_length_img_prop = min(100, subvideo_length) # ensure a minimum of 100 frames for image propagation
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                    pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                    prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                        pred_flows_bi_sub, 
                                                                        masks_dilated[:, s_f:e_f], 
                                                                        'nearest')
                    updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                        prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                    
                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    maybe_empty_cache()

                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                maybe_empty_cache()
        log("image propagation done")
                
        # [T,3,H,W] float32 for blending / repeated averaging
        comp_frames_gpu = torch.zeros((video_length, 3, h, w), device=self.device, dtype=torch.float32)
        comp_filled = torch.zeros((video_length,), device=self.device, dtype=torch.bool)

        neighbor_stride = neighbor_length // 2
        if video_length > subvideo_length:
            ref_num = subvideo_length // ref_stride
        else:
            ref_num = -1
        
        maybe_empty_cache()
        log("start feature propagation + transformer")
        # ---- feature propagation + transformer ----
        for f in tqdm(range(0, video_length, neighbor_stride)):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                    min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
            
            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)
                
                # pred_img = selected_imgs # results of image propagation
                pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)

                ## compose with input frames
                # Keep everything on GPU and do one-time D2H at the end.
                idxs = torch.as_tensor(neighbor_ids, device=self.device, dtype=torch.long)  # [l_t]
                # pred in [0,255]
                pred_u8 = ((pred_img + 1.0) * 127.5).clamp_(0, 255).to(torch.uint8)  # [l_t,3,H,W]
                # original masks in {0,1}
                mask = (masks_dilated_ori[0, idxs, :, :, :] > 0.5).to(torch.float32)  # [l_t,1,H,W]
                mask3 = mask.expand(-1, 3, -1, -1)
                # original frames uint8
                ori_u8 = ori_frames_u8[0].index_select(0, idxs)  # [l_t,3,H,W]
                img = pred_u8.to(torch.float32) * mask3 + ori_u8.to(torch.float32) * (1.0 - mask3)  # [l_t,3,H,W]

                existing = comp_frames_gpu.index_select(0, idxs)
                filled = comp_filled.index_select(0, idxs).view(-1, 1, 1, 1)
                out = torch.where(filled, existing * 0.5 + img * 0.5, img)
                comp_frames_gpu.index_copy_(0, idxs, out)
                comp_filled.index_fill_(0, idxs, True)
            
            maybe_empty_cache()

        # Return GPU tensor priors (avoid any D2H here). Shape: [T,3,H,W] uint8, in RGB.
        comp_u8 = comp_frames_gpu.clamp_(0, 255).to(torch.uint8)  # [T,3,H,W] on CUDA
        del comp_frames_gpu, comp_filled
        maybe_empty_cache()
        log("done")

        if return_priori_frames:
            return comp_u8

        return None



if __name__ == '__main__':

    device = get_device()
    propainter_model_dir = "weights/propainter"
    propainter = Propainter(propainter_model_dir, device=device)

    video = "examples/example1/video.mp4"
    mask =  "examples/example1/mask.mp4"
    output = "results/priori.mp4"
    res = propainter.forward(video, mask, output)
    

    