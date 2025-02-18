import os
import json
import glob
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
import matplotlib.pyplot as plt

FORMAT_INTERLEAVED = True
current_time = datetime.now().strftime("%y%m%d%H%M%S")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def visualize(model, z, file_name):
    x_sample = model.decode_first_stage(z)
    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
    x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
    Image.fromarray(x_sample).save(file_name)

def t2i(model, image_size, prompt, uc, sampler, animal_name, seed=2025, inversion_seed=2025,
        num_frames=8, step=20, scale=7.5, batch_size=8, ddim_eta=0., 
        dtype=torch.float32, device="cuda", camera=None, ddim_depth=35,):
    set_seed(seed)
    if type(prompt)!=list:
        prompt = [prompt]
    
    os.makedirs(f"outputs/{animal_name}_seed{seed}_{current_time}/", exist_ok=True)
    os.makedirs(f"assets/ddim_inv_trajectories_of_renderings/{animal_name}_seed{seed}/", exist_ok=True)
    os.makedirs(f"forward_cache_artefacts/{animal_name}_seed{seed}/reconstruction/", exist_ok=True)
    os.makedirs(f"forward_cache_artefacts/{animal_name}_seed{seed}/articulation/", exist_ok=True)

    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        ### prepare conditions
        # prompt = "a goose flying, wings expanded, 3d asset"
        # prompt = "a black and white cow running, legs bent, font legs bent, back legs bent, 3d asset"
        # prompt = "piggy running, front legs folded at the knees, 3d asset"
        # prompt = "a brown sheep running, legs bent at the knees, front legs bent, back legs bent, 3d asset"
        # prompt = "a gray horse, dark mane on back of its neck, dark tail, running, galloping, front legs bent, back legs bent at the knees, 3d asset"
        # prompt = "a running brown ram, legs bent at the knees, 3d asset"
        c0 = model.get_learned_conditioning(prompt).to(device)
        c1 = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": torch.cat([c0, c1]).repeat(batch_size//2,1,1)}
        # uc = model.get_learned_conditioning("extra tail, missing limbs, bad anatomy").to(device)
        uc = model.get_learned_conditioning("").to(device)
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames
        shape = [num_frames//2, image_size // 8, image_size // 8] # [4, 32, 32]

        ### load saved trajectory
        asset = f"assets/ddim_inv_trajectories_of_renderings/x_inter_rendered_{animal_name}_seed{inversion_seed}.torch"
        sampler.make_schedule(ddim_num_steps=step, ddim_eta=0)

        cached_trajectory = torch.load(asset)   
        if "rendered" in asset:
            cached_trajectory = cached_trajectory[::-1] 
        # now cached_trajectory is from noisiest to cleanest
            
        os.makedirs(f"ablations/{animal_name}/ddim_depth_{ddim_depth}", exist_ok=True)
        # x_T = sampler.stochastic_encode(cached_trajectory[0], torch.tensor([20]).to(device))
        x_T = cached_trajectory[ddim_depth].to(device)
        visualize(model, x_T, f"ablations/{animal_name}/ddim_depth_{ddim_depth}/t2i-starting-point.png")

        ### denoise with supervision from reference frame through rewired self-attention
        samples_ddim, intermediates = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=x_T,
                                        start_time_step=0,
                                        cached_trajectory=asset)
        
        mse_pred_x0 = []
        mse_x_inter = []
        for t, x_t in enumerate(intermediates["pred_x0"]):
            mse = F.mse_loss(x_t[0::2], x_t[1::2]).item()
            mse_pred_x0.append(mse)
            if t == 50:
                visualize(model, x_t, f"ablations/{animal_name}/ddim_depth_{ddim_depth}/pred_x0_t={t:03d}.png")
        
        for t, x_t in enumerate(intermediates["x_inter"]):
            mse = F.mse_loss(x_t[0::2], x_t[1::2]).item()
            mse_x_inter.append(mse)
            if t == 50:
                visualize(model, x_t, f"ablations/{animal_name}/ddim_depth_{ddim_depth}/x_inter_t={t:03d}.png")
        
        plt.figure(figsize=(10, 5))
        plt.plot(mse_pred_x0, label='pred_x0')
        plt.plot(mse_x_inter, label='x_inter')
        plt.xlabel('Time step')
        plt.ylabel('MSE')
        plt.legend()
        plt.title(f'MSE between reference frame and articulation (ddim_depth={ddim_depth})')
        plt.savefig(f"ablations/{animal_name}/ddim_depth_{ddim_depth}/mse_plot.png")
        plt.close()
        
        pngs_to_gif(f"forward_cache_artefacts/{animal_name}_seed{seed}/articulation/", f"outputs/{animal_name}_seed{seed}_{current_time}/forward_articulation_x_inter_{animal_name}_seed{seed}.gif", startswith="x_inter")
        pngs_to_gif(f"forward_cache_artefacts/{animal_name}_seed{seed}/articulation/", f"outputs/{animal_name}_seed{seed}_{current_time}/forward_articulation_pred_x0_{animal_name}_seed{seed}.gif", startswith="pred_x0")
        
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":
    available_animal_assets = os.listdir('assets/renderings/')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="a black and white cow running, legs bent, font legs bent, back legs bent, 3d asset")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--ddim_depth", type=int, default=35, help="ddim_depth")
    parser.add_argument("--num_frames", type=int, default=8, help="num of frames (views) to generate")
    parser.add_argument("--num_rows", type=int, default=1, help="number of rows to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=135)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--inversion_seed", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--animal_name", type=str, default="horse_stallion_highpoly_color_2", 
                        choices=available_animal_assets)
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
    
    t = args.text + args.suffix
    set_seed(args.seed)
    all_img_input_target = []
    input_images = []
    target_images = []
    for j in range(args.num_rows):
        img = t2i(model, args.size, t, uc, sampler, args.animal_name, seed=args.seed, inversion_seed=args.inversion_seed,
                  step=args.step, scale=10, batch_size=batch_size, ddim_eta=0.0, dtype=dtype, device=device, 
                  camera=camera, num_frames=args.num_frames, ddim_depth=args.ddim_depth)
        for i, im in enumerate(img):
            Image.fromarray(im).save(f"outputs/{args.animal_name}_seed{args.seed}_{current_time}/sample_{i}.png")
            if i % 2 == 0:
                input_images.append(im)
            else:
                target_images.append(im)
        img = np.concatenate(img, 1)
        all_img_input_target.append(img)
    all_img_input_target = np.concatenate(all_img_input_target, 0)
    ddim_depth = args.ddim_depth
    animal_name = args.animal_name
    Image.fromarray(all_img_input_target).save(f"ablations/{animal_name}/ddim_depth_{ddim_depth}/sample.png")
    
    # Save all input images together
    input_images_concat = np.concatenate(input_images, axis=1)
    Image.fromarray(input_images_concat).save(f"ablations/{animal_name}/ddim_depth_{ddim_depth}/input_images.png")
    
    # Save all target images together
    target_images_concat = np.concatenate(target_images, axis=1)
    Image.fromarray(target_images_concat).save(f"ablations/{animal_name}/ddim_depth_{ddim_depth}/target_images.png")
    
    # Save input and target images together vertically
    input_target_images_concat = np.concatenate((input_images_concat, target_images_concat), axis=0)
    Image.fromarray(input_target_images_concat).save(f"ablations/{animal_name}/ddim_depth_{ddim_depth}/input_target_images_combined.png")
    
    args.save_json = f"ablations/{animal_name}/ddim_depth_{ddim_depth}/articulation_args.json"
    with open(args.save_json, 'w+') as f:
        json.dump(vars(args), f, indent=4)