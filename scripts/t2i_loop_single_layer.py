import os
import sys
import random
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch 

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):
    if type(prompt)!=list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size,1,1)}
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames
        
        # DDIM sampler schedule is now managed inside the main loop
        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=None)
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="an astronaut riding a horse")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4, help="num of frames (views) to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = max(4, args.num_frames)

    # Loop through each of the 16 layers
    for layer_number in range(16):
        print(f"\n{'='*20} Generating for Layer {layer_number} {'='*20}")
        
        # --- Create rewire_switch for the current layer ---
        rewire_switch = [0] * 16
        rewire_switch[layer_number] = 1
        print(f"Using rewire_switch: {rewire_switch}")

        config_override = OmegaConf.create({
            "model": {
                "params": {
                    "unet_config": {
                        "params": {
                            "rewire_switch": rewire_switch
                        }
                    }
                }
            }
        })
        
        # --- Load the model inside the loop to apply the new config ---
        print("Loading t2i model...")
        if args.config_path is None:
            model = build_model(args.model_name, ckpt_path=args.ckpt_path, config_overrides=config_override)
        else:
            assert args.ckpt_path is not None, "ckpt_path must be specified!"
            config = OmegaConf.load(args.config_path)
            # Apply override to loaded config
            config.merge_with(config_override)
            model = instantiate_from_config(config.model)
            model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
        
        model.device = device
        model.to(device)
        model.eval()

        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=50, ddim_eta=0, verbose=False)
        uc = model.get_learned_conditioning([""]).to(device)
        print("Load t2i model done.")

        # --- Pre-compute camera matrices ---
        if args.use_camera:
            camera = get_camera(args.num_frames, elevation=args.camera_elev, 
                    azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
            camera = camera.repeat(batch_size//args.num_frames,1).to(device)
        else:
            camera = None
        
        t = args.text + args.suffix
        set_seed(args.seed)
        
        # --- Generate Image ---
        # Note: The original script generated 3 rows. For simplicity, this version generates one.
        # You can re-introduce the inner loop if you need multiple rows per layer.
        img = t2i(model, args.size, t, uc, sampler, step=50, scale=10, batch_size=batch_size, ddim_eta=0.0, 
                  dtype=dtype, device=device, camera=camera, num_frames=args.num_frames)
        img = np.concatenate(img, 1)

        # --- Save Image with layer-specific name ---
        output_filename = f"sample_{layer_number}.png"
        Image.fromarray(img).save(output_filename)
        print(f"Saved output to {output_filename}")
        
        # Clean up model to free GPU memory for the next iteration
        del model
        del sampler
        torch.cuda.empty_cache()