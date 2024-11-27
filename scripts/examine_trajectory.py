import os
import re
import json
import random
from datetime import datetime
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch 
import torch.nn.functional as F
from tqdm import tqdm
from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from torchvision.transforms.functional import to_tensor
from pngs2gif import pngs_to_gif

FORMAT_INTERLEAVED = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def t2i(model, image_size, prompt, uc, sampler,
        step=50, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, 
        device="cuda", camera=None, num_frames=4):
    current_time = datetime.now().strftime("%y%m%d%H%M%S")
    folder_name = re.sub(r'\W', '_', prompt)
    if type(prompt)!=list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c0 = model.get_learned_conditioning(prompt).to(device)
        c1 = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": torch.cat([c0, c1]).repeat(batch_size//2,1,1)}
        uc = model.get_learned_conditioning("").to(device)
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames
        shape = [4, image_size // 8, image_size // 8] # [4, 32, 32]

        os.makedirs(f"outputs/{folder_name}_seed{args.seed}_{current_time}/", exist_ok=True)
        os.makedirs(f"forward_cache_artefacts/{folder_name}_seed{args.seed}/reconstruction/", exist_ok=True)
        os.makedirs(f"forward_cache_artefacts/{folder_name}_seed{args.seed}/articulation/", exist_ok=True)
        
        samples_ddim, intermediates = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=None,
                                        start_time_step=0,)
        for t, x_t in enumerate(intermediates["pred_x0"]):
            x_sample = model.decode_first_stage(x_t)
            print(t, ",", F.mse_loss(x_t, intermediates["pred_x0"][-1]), ",", F.mse_loss(x_sample, model.decode_first_stage(intermediates["pred_x0"][-1])))
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
            x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
            Image.fromarray(x_sample).save(f"forward_cache_artefacts/{folder_name}_seed{args.seed}/reconstruction/pred_x0_t={t:03d}.png")
            
        for t, x_t in enumerate(intermediates["x_inter"]):
            x_sample = model.decode_first_stage(x_t)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
            x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
            Image.fromarray(x_sample).save(f"forward_cache_artefacts/{folder_name}_seed{args.seed}/reconstruction/x_inter_t={t:03d}.png")
        pngs_to_gif(f"forward_cache_artefacts/{folder_name}_seed{args.seed}/reconstruction/", f"outputs/{folder_name}_seed{args.seed}_{current_time}/forward_reconstruction_x_inter_{folder_name}_seed{args.seed}.gif", startswith="x_inter")
        pngs_to_gif(f"forward_cache_artefacts/{folder_name}_seed{args.seed}/reconstruction/", f"outputs/{folder_name}_seed{args.seed}_{current_time}/forward_reconstruction_pred_x0_{folder_name}_seed{args.seed}.gif", startswith="pred_x0")
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v1.5-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="Homer Simpson")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=8, help="num of frames (views) to generate")
    parser.add_argument("--num_rows", type=int, default=1, help="number of rows to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=135)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = args.num_frames

    print("load t2i model ... ")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"))
    model.device = device
    model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning( [""] ).to(device)
    print("loaded t2i model. ")

    # pre-compute camera matrices
    if args.use_camera:
        camera = get_camera(args.num_frames//2, elevation=args.camera_elev, 
                azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
        if FORMAT_INTERLEAVED:
            camera = camera.repeat_interleave(2*batch_size//args.num_frames,dim=0).to(device)    # AABBCCDD
        else:
            camera = camera.repeat(2*batch_size//args.num_frames,1).to(device)                   # ABCDABCD
        print(camera.shape)
    else:
        camera = None
    
    folder_name = re.sub(r'\W', '_', args.text)
    os.makedirs(f"outputs/{folder_name}_seed{args.seed}")
    t = args.text + args.suffix
    set_seed(args.seed)
    images = []
    for j in range(args.num_rows):
        img = t2i(model, args.size, t, uc, sampler, step=args.step, scale=10, batch_size=batch_size, ddim_eta=0.0, 
                dtype=dtype, device=device, camera=camera, num_frames=args.num_frames,)
        for i, im in enumerate(img):
            Image.fromarray(im).save(f"outputs/{folder_name}_seed{args.seed}_{current_time}/sample_{i}.png")
        img = np.concatenate(img, 1)
        images.append(img)
    images = np.concatenate(images, 0)
    Image.fromarray(images).save(f"outputs/{folder_name}_seed{args.seed}_{current_time}/sample.png")
    args.save_json = f"outputs/{folder_name}_seed{args.seed}_{current_time}/inversion_args.json"
    with open(args.save_json, 'w+') as f:
        json.dump(vars(args), f, indent=4)