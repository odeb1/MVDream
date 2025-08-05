import time
import os, sys
sys.path = [p for p in sys.path if "MVDream" not in p]
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, os.path.join(current_dir, ".."))

import json
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import torch 
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from pngs2gif import pngs_to_gif

FORMAT_INTERLEAVED = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

brightness = random.uniform(0.9, 1.1)  # Adjust brightness by a factor between 0.5 and 1.5
contrast = random.uniform(0.7, 1.2)    # Adjust contrast by a factor between 0.5 and 1.5
saturation = random.uniform(0.7, 1.0)  # Adjust saturation by a factor between 0.5 and 1.5
hue = random.uniform(-0.1, 0.1)        # Adjust hue by a factor between -0.1 and 0.1

def t2i(model, image_size, prompt, uc, sampler, animal_name, 
        step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, 
        device="cuda", camera=None, num_frames=1, start_time_step=35):

    output_dir = f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}/"
    os.makedirs(output_dir, exist_ok=True)

    # import ipdb; ipdb.set_trace()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        # prompt = "a tiger standing straight, legs straight, animal, 3d asset"
        # prompt = "a brown sheep, myanmar, nyilonelycompany, standing straight, legs straight, white face, fluffy, 3d asset"
        # prompt = "a cow standing, 3D asset"
        # prompt = "a gray horse, dark mane on back of its neck, dark tail, standing straight, 3d asset"
        # prompt = "a dark brown ram with curled horns, reddish eyes, textured wool coat, standing upright, 3d asset"

        c0 = model.get_learned_conditioning(prompt).to(device)
        c1 = model.get_learned_conditioning(prompt).to(device)
        # The two embeddings are concatenated and then repeated to match the batch_size. 
        # The division by 2 suggests that two prompt embeddings are used per pair of items in the batch.
        c_ = {"context": torch.cat([c0, c1]).repeat(batch_size//2,1,1)}
        ##### for vf, context in enumerate(c_["context"]):  print(vf, context.shape, context[4:16])
        # uc = model.get_learned_conditioning("resting, wings tucked in, wings folded tightly against its sides").to(device)
        uc = model.get_learned_conditioning("").to(device)
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        
        # if camera data exists, this code adds it to both the conditional (c_) and unconditional (uc_) context dictionaries.
        # The model needs to know the viewpoint for every step of the diffusion process. By providing the camera data to both contexts, 
        # we ensure the spatial information is always available, whether the model is referring to the text prompt or not.
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames
        
        # Defines a list "shape" of the latent space tensor. image_size // 8 is common because many diffusion models 
        # (like Stable Diffusion) use an autoencoder (VAE) that downsamples the image by a factor of 8. 
        # args.num_frames//2 suggests the number of distinct views being processed in this latent batch.
        shape = [args.num_frames//2, image_size // 8, image_size // 8] # [4, 32, 32]

        os.makedirs(f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}/", exist_ok=True)
        os.makedirs(f"{args.folder_path_save}/assets/ddim_inv_trajectories_of_renderings/{animal_name}_seed{args.seed}/", exist_ok=True)
        os.makedirs(f"{args.folder_path_save}/forward_cache_artefacts/{animal_name}_seed{args.seed}/reconstruction/", exist_ok=True)
        os.makedirs(f"{args.folder_path_save}/forward_cache_artefacts/{animal_name}_seed{args.seed}/articulation/", exist_ok=True)
        
        x = []
        azimuthal_interval = 360 // (args.num_frames//2)
        print(azimuthal_interval, "azimuthal_interval")
        for i in range(args.num_frames//2):
            # Load the image
            a = animal_name.split("_")[0]
            image = Image.open(os.path.join(f"assets/renderings/{animal_name}/", f"{a}_{45+azimuthal_interval*i:03d}.png")).resize((512,512)).convert("RGB")
            
            # Apply the transformations
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)
            # image = TF.adjust_hue(image, hue)

            x.append(to_tensor(image))
        
        # Normalizes pixel values from [0, 1] to [-1, 1]
        x = torch.stack(x).to(device) * 2.0 - 1.0
        x = F.interpolate(x, (256, 256))
        
        # VAE encoder
        x_T = model.encode_first_stage(x).mean #.sample() # DiagonalGaussianDistribution
        # Normalising the output (latent distribution) from the VAE encoder
        x_T = x_T * 0.18215 # IMPORTANT!!

        if FORMAT_INTERLEAVED:
            x_T = x_T.repeat_interleave(2, dim=0)    # AABBCCDD
        else:
            repeater = num_frames // 4
            x_T = x_T.repeat(repeater, 1, 1, 1)      # ABCDABCD
            
        print(x_T.shape, c_["context"].shape, uc_["context"].shape, shape, batch_size)
        # torch.Size([n_frames, 4, 32, 32]) torch.Size([n_frames, 77, 1024]) torch.Size([n_frames, 77, 1024]) [n_frames//2, 32, 32] n_frames
        samples, intermediates_inversion = sampler.sample_inversion(S=step, conditioning=c_,
                                    batch_size=batch_size, shape=shape,
                                    verbose=False, 
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc_,
                                    eta=ddim_eta, x_T=x_T,)
        print(len(intermediates_inversion["x_inter"]), intermediates_inversion["x_inter"][0].shape) 
        x_T = intermediates_inversion["x_inter"][-(start_time_step+1)]
        x_T[1::2] = torch.randn_like(x_T[1::2])
        print(x_T[1::2].shape)
        for t, x_t in enumerate(intermediates_inversion["x_inter"]):
            x_sample = model.decode_first_stage(x_t)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
            x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
            Image.fromarray(x_sample).save(f"{args.folder_path_save}/assets/ddim_inv_trajectories_of_renderings/{animal_name}_seed{args.seed}/x_inter_t={t:03d}.png")
        pngs_to_gif(f"{args.folder_path_save}/assets/ddim_inv_trajectories_of_renderings/{animal_name}_seed{args.seed}/", f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}/ddim_inv_trajectory_of_renderings_{animal_name}_seed{args.seed}.gif")
        asset = f"{args.folder_path_save}/assets/ddim_inv_trajectories_of_renderings/x_inter_rendered_{animal_name}_seed{args.seed}.torch"
        torch.save(intermediates_inversion["x_inter"], asset)

        noise_diff_norm = [d.norm(p=2) for d in [(tn - un) for tn, un in zip(intermediates_inversion["model_t"], intermediates_inversion["model_uncond"])]]
        noise_diff_norm = torch.stack(noise_diff_norm, dim=0).cpu().numpy()
        print("Noise diff norms", noise_diff_norm.shape)
        plt.figure(figsize=(10, 6))
        steps = list(reversed(range(len(noise_diff_norm))))
        plt.plot(steps, noise_diff_norm, 'b-', linewidth=2)
        plt.xlabel('DDIM Step')
        plt.ylabel('Noise Diff Norm')
        plt.title(f'Noise Difference Norm vs DDIM Step - {animal_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/noise_diff_norm_inver.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Inversion Noise Diff Norm plot saved at:", os.path.join(output_dir, "noise_diff_norm_inver.png"))


        x_T = intermediates_inversion["x_inter"][25].to(device)
        # x_T = sampler.stochastic_encode(intermediates_inversion["x_inter"][0], torch.tensor([25]).to(device))
        # x_T[1::2] = torch.randn_like(x_T_resampled[1::2])
        samples_ddim, intermediates = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=x_T,
                                        start_time_step=0,)

        noise_diff_norm = [d.norm(p=2) for d in [(tn - un) for tn, un in zip(intermediates["model_t"], intermediates["model_uncond"])]]
        noise_diff_norm = torch.stack(noise_diff_norm, dim=0).cpu().numpy()
        print("Noise diff norms", noise_diff_norm.shape)
        plt.figure(figsize=(10, 6))
        steps = list(range(len(noise_diff_norm)))
        plt.plot(steps, noise_diff_norm, 'b-', linewidth=2)
        plt.xlabel('DDIM Step')
        plt.ylabel('Noise Diff Norm')
        plt.title(f'Noise Difference Norm vs DDIM Step - {animal_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/noise_diff_norm_recon.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Reconstruction Noise Diff Norm plot saved at:", os.path.join(output_dir, "noise_diff_norm_recon.png"))

        # csv_path = os.path.join(output_dir, "mse.csv")
        # if not os.path.exists(csv_path):
        #     with open(csv_path, 'w') as f:
        #         f.write("t, mse_x_t, mse_x_sample\n")

        for t, x_t in enumerate(intermediates["pred_x0"]):
            x_sample = model.decode_first_stage(x_t)
            # mse_x_t = F.mse_loss(x_t, intermediates_inversion["pred_x0"][0]).item()
            # mse_x_sample = F.mse_loss(x_sample, model.decode_first_stage(intermediates["pred_x0"][0])).item()
            ### Print and write to file
            # print(t, ",", mse_x_t, ",", mse_x_sample)
            # with open(csv_path, 'a') as f:
            #     f.write(f"{t}, {mse_x_t}, {mse_x_sample}\n")

            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
            x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
            Image.fromarray(x_sample).save(f"{args.folder_path_save}/forward_cache_artefacts/{animal_name}_seed{args.seed}/reconstruction/pred_x0_t={t:03d}.png")
        for t, x_t in enumerate(intermediates["x_inter"]):
            x_sample = model.decode_first_stage(x_t)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
            x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
            Image.fromarray(x_sample).save(f"{args.folder_path_save}/forward_cache_artefacts/{animal_name}_seed{args.seed}/reconstruction/x_inter_t={t:03d}.png")
        pngs_to_gif(f"{args.folder_path_save}/forward_cache_artefacts/{animal_name}_seed{args.seed}/reconstruction/", f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}/forward_reconstruction_x_inter_{animal_name}_seed{args.seed}.gif", startswith="x_inter")
        pngs_to_gif(f"{args.folder_path_save}/forward_cache_artefacts/{animal_name}_seed{args.seed}/reconstruction/", f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}/forward_reconstruction_pred_x0_{animal_name}_seed{args.seed}.gif", startswith="pred_x0")
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample[::2] = x # replace the first frame by the original image
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":
    available_animal_assets = os.listdir('assets/renderings/')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="a brown horse standing still, legs straight")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--step", type=int, default=50)              # Keep it 50 IMPORTANT
    parser.add_argument("--start_time_step", type=int, default=30, help="DDIM inversion start time step")
    parser.add_argument("--num_frames", type=int, default=8, help="num of frames (views) to generate")
    parser.add_argument("--num_rows", type=int, default=1, help="number of rows to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=135)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--animal_name", type=str, default="horse_stallion_highpoly_color_2",
                        choices=available_animal_assets)
    parser.add_argument("--folder_path_save", type=str, default="../results", help="folder_path")
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
    
    start_time = time.time()
    
    t = args.text + args.suffix
    set_seed(args.seed)
    images = []
    for j in range(args.num_rows):
        img = t2i(model, args.size, t, uc, sampler, args.animal_name, step=args.step, scale=10, batch_size=batch_size, ddim_eta=0.0, 
                dtype=dtype, device=device, camera=camera, num_frames=args.num_frames, start_time_step=args.start_time_step)
        # for i, im in enumerate(img):
        #     Image.fromarray(im).save(f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}/sample_{i}.png")
        img = np.concatenate(img, 1)
        images.append(img)
    
    
    end_time = time.time()  # Record the end time 
    print(f"DDIM Inversion took {end_time - start_time} seconds to run.")
    
    images = np.concatenate(images, 0)
    Image.fromarray(images).save(f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}/reconstruction.png")

    args.save_json = f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}/inversion_args.json"
    with open(args.save_json, 'w+') as f:
        json.dump(vars(args), f, indent=4)