import time
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
    
def numpy_to_python(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    return obj

def save_metrics_json(metrics_dict, filepath):
    """Safely save metrics dictionary to JSON file."""
    # Convert all numpy values to Python types
    json_safe_metrics = {}
    for depth, metrics in metrics_dict.items():
        json_safe_metrics[str(depth)] = {  # Convert depth to string for JSON
            k: numpy_to_python(v) for k, v in metrics.items()
        }
    
    with open(filepath, 'w') as f:
        json.dump(json_safe_metrics, f, indent=4)
        
def t2i(model, image_size, prompt, uc, sampler, animal_name, seed=2025, inversion_seed=2025,
        num_frames=8, step=20, scale=7.5, batch_size=8, ddim_eta=0., 
        dtype=torch.float32, device="cuda", camera=None, ddim_depth=35, use_ddim_inversion=False):
    set_seed(seed)
    if type(prompt)!=list:
        prompt = [prompt]
    
    os.makedirs(f"{args.folder_path_save}/outputs/{animal_name}_seed{seed}_{current_time}/", exist_ok=True)
    os.makedirs(f"{args.folder_path_save}/assets/ddim_inv_trajectories_of_renderings/{animal_name}_seed{seed}_{current_time}/", exist_ok=True)
    os.makedirs(f"{args.folder_path_save}/forward_cache_artefacts/{animal_name}_seed{seed}_{current_time}/reconstruction/", exist_ok=True)
    os.makedirs(f"{args.folder_path_save}/forward_cache_artefacts/{animal_name}_seed{seed}_{current_time}/articulation/", exist_ok=True)

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

        x_T = None # Initialize x_T to None
        asset = None
        
        if args.use_ddim_inversion:
            print("DDIM Inversion is ON. Loading from cached trajectory.")
            ### load saved trajectory
            asset = f"{args.folder_path_save}/assets/ddim_inv_trajectories_of_renderings/x_inter_rendered_{animal_name}_seed{inversion_seed}.torch"
            sampler.make_schedule(ddim_num_steps=step, ddim_eta=0)

            try:
                cached_trajectory = torch.load(asset)
            except FileNotFoundError:
                print(f"ERROR: DDIM inversion asset not found at {asset}")
                print("Please generate the inversion trajectory first or run with DDIM inversion turned off.")
                # Return an empty list or raise an error to stop execution
                return [], [], []
            
            if "rendered" in asset:
                cached_trajectory = cached_trajectory[::-1] 
            # now cached_trajectory is from noisiest to cleanest

            os.makedirs(f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}_{current_time}/ddim_depth_{ddim_depth}", exist_ok=True)
            # x_T = sampler.stochastic_encode(cached_trajectory[0], torch.tensor([20]).to(device))
            x_T = cached_trajectory[ddim_depth].to(device)
            visualize(model, x_T, f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}_{current_time}/ddim_depth_{ddim_depth}/ddim_depth_{ddim_depth}_t2i-starting-point.png")
        else:
            print("DDIM Inversion is OFF. Starting from random noise.")
            # When x_T is None, the DDIMSampler will automatically start from random noise.


        ### denoise with supervision from reference frame through rewired self-attention
        samples_ddim, intermediates = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=x_T,
                                        start_time_step=0,
                                        cached_trajectory=asset)
        output_dir = f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}_{current_time}/ddim_depth_{ddim_depth}/"
        noise_diff_norm = [d.norm(p=2) for d in [(tn - un) for tn, un in zip(intermediates["model_t"], intermediates["model_uncond"])]]
        noise_diff_norm = torch.stack(noise_diff_norm, dim=0).cpu().numpy()
        print("Articulation - Noise Diff Norms", noise_diff_norm.shape)
        plt.figure(figsize=(10, 6))
        steps = list(range(len(noise_diff_norm)))
        plt.plot(steps, noise_diff_norm, 'b-', linewidth=2)
        plt.xlabel('DDIM Step')
        plt.ylabel('Noise Diff Norm')
        plt.title(f'Articulation - DDIM_Depth_{ddim_depth} - Noise Difference Norm vs DDIM Step - {animal_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/noise_diff_norm_recon.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Reconstruction Noise Diff Norm plot saved at:", os.path.join(output_dir, f"ddim_depth_{ddim_depth}_noise_diff_norm_artic.png"))

        mse_pred_x0 = []
        mse_x_inter = []
        target = intermediates["pred_x0"][-1][0::2]
        for t, x_t in enumerate(intermediates["pred_x0"]):
            mse = F.mse_loss(model.decode_first_stage(target), model.decode_first_stage(x_t[1::2])).item()
            mse_pred_x0.append(mse)
            if t == 50:
                visualize(model, x_t, f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}_{current_time}/ddim_depth_{ddim_depth}/ddim_depth_{ddim_depth}_pred_x0_t={t:03d}.png")
        
        for t, x_t in enumerate(intermediates["x_inter"]):
            mse = F.mse_loss(model.decode_first_stage(target), model.decode_first_stage(x_t[1::2])).item()
            mse_x_inter.append(mse)
            if t == 50:
                visualize(model, x_t, f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}_{current_time}/ddim_depth_{ddim_depth}/ddim_depth_{ddim_depth}_x_inter_t={t:03d}.png")
        
        plt.figure(figsize=(10, 5))
        plt.plot(mse_pred_x0, label='pred_x0')
        plt.plot(mse_x_inter, label='x_inter')
        plt.xlabel('Time step')
        plt.ylabel('MSE')
        plt.legend()
        plt.title(f'MSE between reference frame and articulation (ddim_depth={ddim_depth})')
        plt.savefig(f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}_{current_time}/ddim_depth_{ddim_depth}/ddim_depth_{ddim_depth}_mse_plot.png")
        plt.close()
        
        pngs_to_gif(f"{args.folder_path_save}/forward_cache_artefacts/{animal_name}_seed{seed}_{current_time}/articulation/", f"{args.folder_path_save}/outputs/{animal_name}_seed{seed}_{current_time}/forward_articulation_x_inter_{animal_name}_seed{seed}.gif", startswith="x_inter")
        pngs_to_gif(f"{args.folder_path_save}/forward_cache_artefacts/{animal_name}_seed{seed}_{current_time}/articulation/", f"{args.folder_path_save}/outputs/{animal_name}_seed{seed}_{current_time}/forward_articulation_pred_x0_{animal_name}_seed{seed}.gif", startswith="pred_x0")
        
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8)), mse_pred_x0, mse_x_inter


def run_forward_inference(args):
    print("Running forward inference...")
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
    all_img_input_target = []
    input_images = []
    target_images = []
    for j in range(args.num_rows):
        img, _, _ = t2i(model, args.size, t, uc, sampler, args.animal_name, seed=args.seed, inversion_seed=args.inversion_seed,
                  step=args.step, scale=10, batch_size=batch_size, ddim_eta=0.0, dtype=dtype, device=device, 
                  camera=camera, num_frames=args.num_frames, ddim_depth=args.ddim_depth, use_ddim_inversion=args.use_ddim_inversion)
        for i, im in enumerate(img):
            Image.fromarray(im).save(f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}_{current_time}/ddim_depth_{args.ddim_depth}/sample_{i}.png")
            if i % 2 == 0:
                input_images.append(im)
                Image.fromarray(im).save(f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}_{current_time}/{args.animal_name}_ddim_depth_{args.ddim_depth}_f0v{i//2}.png")
            else:
                target_images.append(im)
                Image.fromarray(im).save(f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}_{current_time}/{args.animal_name}_ddim_depth_{args.ddim_depth}_f1v{i//2}.png")
        img = np.concatenate(img, 1)
        all_img_input_target.append(img)
    
    end_time = time.time()  # Record the end time 
    print(f"Articulation.py took {end_time - start_time} seconds to run.")
    
    all_img_input_target = np.concatenate(all_img_input_target, 0)

    Image.fromarray(all_img_input_target).save(f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}_{current_time}/{args.animal_name}_ddim_depth_{args.ddim_depth}_sample.png")
    
    # Save all input images together
    input_images_concat = np.concatenate(input_images, axis=1)
    Image.fromarray(input_images_concat).save(f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}_{current_time}/{args.animal_name}_ddim_depth_{args.ddim_depth}_input_images.png")
    
    # Save all target images together
    target_images_concat = np.concatenate(target_images, axis=1)
    Image.fromarray(target_images_concat).save(f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}_{current_time}/{args.animal_name}_ddim_depth_{args.ddim_depth}_target_images.png")
    
    # Save input and target images together vertically
    input_target_images_concat = np.concatenate((input_images_concat, target_images_concat), axis=0)
    Image.fromarray(input_target_images_concat).save(f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}_{current_time}/{args.animal_name}_ddim_depth_{args.ddim_depth}_input_target_images_combined.png")
    
    args.save_json = f"{args.folder_path_save}/outputs/{args.animal_name}_seed{args.seed}_{current_time}/articulation_args.json"
    with open(args.save_json, 'w+') as f:
        json.dump(vars(args), f, indent=4)


def run_ddim_depth_ablation(args):
    """Run ablation study for different DDIM depths and automatically select the best one."""
    print("Run ablation study for different DDIM depths and automatically select the best one.")
    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = args.num_frames

    print("Loading t2i model...")
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
    uc = model.get_learned_conditioning([""] ).to(device)
    
    # Pre-compute camera matrices
    camera = None
    if args.use_camera:
        camera = get_camera(args.num_frames//2, elevation=args.camera_elev, 
                          azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
        if FORMAT_INTERLEAVED:
            camera = camera.repeat_interleave(2*batch_size//args.num_frames,dim=0).to(device)
        else:
            camera = camera.repeat(2*batch_size//args.num_frames,1).to(device)
    
    # Dictionary to store metrics for each depth
    depth_metrics = {}
    
    # for_loop
    # Try different DDIM depths
    for ddim_depth in range(5, 50, 5):
        print(f"\nTesting DDIM depth: {ddim_depth}")
        
        # Create directories for this depth
        ddim_depth_folder = f"{args.folder_path_save}/ablations_ddim_depth/{args.animal_name}/ddim_depth_{ddim_depth}"
        os.makedirs(ddim_depth_folder, exist_ok=True)
        
        # Run generation with current depth
        t = args.text + args.suffix
        set_seed(args.seed)
        
        # Generate images with current depth setting
        img, mse_pred_x0, mse_x_inter = t2i(model, args.size, t, uc, sampler, args.animal_name, 
                 seed=args.seed, inversion_seed=args.inversion_seed,
                 step=args.step, scale=10, batch_size=batch_size, 
                 ddim_eta=0.0, dtype=dtype, device=device,
                 camera=camera, num_frames=args.num_frames, 
                 ddim_depth=ddim_depth,
                 use_ddim_inversion=args.use_ddim_inversion)
        
        # Save the results for this depth
        input_images = []
        target_images = []
        
        for i, im in enumerate(img):
            Image.fromarray(im).save(f"{args.folder_path_save}/ablations_ddim_depth/{args.animal_name}/ddim_depth_{ddim_depth}/sample_{i}.png")
            if i % 2 == 0:
                input_images.append(im)
            else:
                target_images.append(im)
        
        
        # Save combined visualizations
        input_images_concat = np.concatenate(input_images, axis=1)
        target_images_concat = np.concatenate(target_images, axis=1)
        
        # Save input and target images individually
        Image.fromarray(input_images_concat).save(f"{args.folder_path_save}/ablations_ddim_depth/{args.animal_name}/ddim_depth_{ddim_depth}/input_images.png")
        Image.fromarray(target_images_concat).save(f"{args.folder_path_save}/ablations_ddim_depth/{args.animal_name}/ddim_depth_{ddim_depth}/target_images.png")
        
        input_target_combined = np.concatenate((input_images_concat, target_images_concat), axis=0)
        Image.fromarray(input_target_combined).save(os.path.join(ddim_depth_folder, "input_target_combined.png"))

        # Calculate metrics for curve comparison
        def calculate_curve_metrics(pred_x0_vals, x_inter_vals):
            # Convert to numpy arrays for easier calculation
            pred_x0_arr = np.array(pred_x0_vals)
            x_inter_arr = np.array(x_inter_vals)
            
            # Calculate absolute differences between curves
            differences = np.abs(pred_x0_arr - x_inter_arr)
            
            return {
                'mean_difference': float(np.mean(differences)),
                'max_difference': float(np.max(differences)),
                'std_difference': float(np.std(differences)),
                'convergence_point': int(np.argmin(differences)),
                'stable_region_size': int(np.sum(differences < 0.1)),
                'final_difference': float(differences[-1])
            }
        
        metrics = calculate_curve_metrics(mse_pred_x0, mse_x_inter)
        depth_metrics[ddim_depth] = metrics
        print(depth_metrics)
        
        with open(f"{args.folder_path_save}/ablations_ddim_depth/{args.animal_name}/ddim_depth_{ddim_depth}/metrics.json", 'w') as f:
            json.dump({
                'mse_pred_x0': numpy_to_python(mse_pred_x0),
                'mse_x_inter': numpy_to_python(mse_x_inter),
                'metrics': metrics  # metrics are already Python types from calculate_curve_metrics
            }, f, indent=4)
    
    # Select optimal depth based on metrics
    def select_optimal_depth(metrics_dict):
        scores = {}
        for depth, metrics in metrics_dict.items():
            # Calculate a composite score (lower is better)
            score = (
                metrics['mean_difference'] * 0.3 +  # Weight for average difference
                metrics['std_difference'] * 0.2 +   # Weight for stability
                (1.0 / (metrics['stable_region_size'] + 1)) * 0.3 +  # Weight for convergence
                metrics['final_difference'] * 0.2    # Weight for final convergence
            )
            scores[depth] = score
            
        # Find the depth with minimum score
        optimal_depth = min(scores.items(), key=lambda x: x[1])[0]
        
        # Check for sharp transitions after optimal depth
        depths = sorted(list(metrics_dict.keys()))
        optimal_idx = depths.index(optimal_depth)
        
        # If we're not at the last depth, check for sharp transitions
        if optimal_idx < len(depths) - 1:
            next_depth = depths[optimal_idx + 1]
            score_increase = scores[next_depth] / scores[optimal_depth]
            
            # If score gets significantly worse (>50% increase), stick with current depth
            if score_increase > 1.5:
                return optimal_depth
            # If next depth isn't much worse, consider using it
            elif score_increase < 1.2:
                return next_depth
        
        return optimal_depth
    
    optimal_depth = select_optimal_depth(depth_metrics)
    
    # Save analysis results
    analysis_results = {
        'optimal_depth': optimal_depth,
        'depth_metrics': depth_metrics,
        'selection_criteria': {
            'description': 'Composite score based on mean difference, stability, convergence region, and final difference',
            'weights': {
                'mean_difference': 0.3,
                'std_difference': 0.2,
                'stable_region_size': 0.3,
                'final_difference': 0.2
            }
        }
    }
    with open(f"{args.folder_path_save}/ablations_ddim_depth/{args.animal_name}/ddim_depth_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    print(f"\nOptimal DDIM depth selected: {optimal_depth}")
    print(f"Analysis results saved to: ablations_ddim_depth/{args.animal_name}/ddim_depth_analysis.json")
    
    return optimal_depth


def _run_single_configuration(args, rewire_switch, config_name, base_output_dir):
    """Helper function to run the generation and saving process for a single configuration."""
    print(f"--- Running configuration: {config_name} ---")
    
    # Create config override
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

    # Create output directory for this configuration
    iteration_dir = os.path.join(base_output_dir, config_name)
    os.makedirs(iteration_dir, exist_ok=True)

    # Build model with this configuration
    model = build_model(args.model_name, ckpt_path=args.ckpt_path, config_overrides=config_override)
    model.device = args.device
    model.to(args.device)
    model.eval()

    # Save configuration
    with open(os.path.join(iteration_dir, "rewire_config.json"), 'w') as f:
        json.dump({"rewire_switch": rewire_switch}, f, indent=4)


    # Run generation with this configuration
    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning([""]).to(args.device)

    # Set up camera if needed
    camera = get_camera(args.num_frames//2, elevation=args.camera_elev,
                      azimuth_start=args.camera_azim, 
                      azimuth_span=args.camera_azim_span) if args.use_camera else None
    if camera is not None:
        if FORMAT_INTERLEAVED:
            camera = camera.repeat_interleave(2*args.num_frames//args.num_frames,dim=0).to(args.device)
        else:
            camera = camera.repeat(2*args.num_frames//args.num_frames,1).to(args.device)

    # Generate images
    t = args.text + args.suffix
    images, _, _ = t2i(model, args.size, t, uc, sampler, args.animal_name,
                seed=args.seed, inversion_seed=args.inversion_seed,
                step=args.step, scale=10, batch_size=args.num_frames,
                ddim_eta=0.0, dtype=torch.float32 if not args.fp16 else torch.float16,
                device=args.device, camera=camera, num_frames=args.num_frames,
                ddim_depth=args.ddim_depth,
                use_ddim_inversion=args.use_ddim_inversion)

    # Save results
    input_images = []
    target_images = []
    for idx, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(iteration_dir, f"sample_{idx}.png"))
        if idx % 2 == 0:
            input_images.append(img)
        else:
            target_images.append(img)

    # Save combined visualizations
    input_images_concat = np.concatenate(input_images, axis=1)
    target_images_concat = np.concatenate(target_images, axis=1)
    input_target_combined = np.concatenate((input_images_concat, target_images_concat), axis=0)
    Image.fromarray(input_target_combined).save(os.path.join(iteration_dir, "input_target_combined.png"))
    
    Image.fromarray(input_images_concat).save(os.path.join(iteration_dir, "input_images.png"))
    Image.fromarray(target_images_concat).save(os.path.join(iteration_dir, "target_images.png"))
    
    print(f"--- Finished configuration: {config_name} ---")

def create_switch(*ranges):
    """Creates a 16-element boolean list, setting specified index ranges to True."""
    switch = [False] * 16
    for start, end in ranges:
        for i in range(start, end + 1):
            switch[i] = True
    return switch

def run_rewire_switch_ablation(args):
    """Run ablation study for different rewire switch configurations"""
    print("Run ablation study for different rewire switch configurations.")
    base_output_dir = f"{args.folder_path_save}/ablations_rewire_switch/{args.animal_name}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # --- Define all configurations to be tested ---
    configurations = {}

    # 1. 16 iterations with only one True value
    for i in range(16):
        configurations[f"config_only_{i}_true"] = create_switch((i, i))

    # 2. All 16 are False
    configurations["config_all_false"] = [False] * 16
    
    # 3. All 16 are True (Corrected from original code)
    configurations["config_all_true"] = [True] * 16

    # 4. Only 0 to 7 are True
    configurations["config_0_to_7_true"] = create_switch((0, 7))
    
    # 5. Only 8 to 15 are True
    configurations["config_8_to_15_true"] = create_switch((8, 15))

    # --- Check for the flag to run additional, more complex ablations ---
    if args.run_more_rsa_ablations:
        print("--- 'run_more_ablations' flag is set. Adding more configurations. ---")
        more_configs = {
            "config_0_to_3_true": create_switch((0, 3)),
            "config_4_to_7_true": create_switch((4, 7)),
            "config_8_to_10_true": create_switch((8, 10)),
            "config_10_to_15_true": create_switch((10, 15)),
            "config_0_to_3_and_10_to_15_true": create_switch((0, 3), (10, 15)),
            "config_4_to_7_and_8_to_10_true": create_switch((4, 7), (8, 10)),
            "config_4_to_7_and_10_to_15_true": create_switch((4, 7), (10, 15)),
        }
        configurations.update(more_configs)

    # --- Run the experiments for each defined configuration ---
    for name, switch_config in configurations.items():
        _run_single_configuration(
            args=args,
            rewire_switch=switch_config,
            config_name=name,
            base_output_dir=base_output_dir
        )



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
    parser.add_argument("--run_ddim_depth_ablation", action="store_true")
    parser.add_argument("--run_rewire_switch_ablation", action="store_true")
    parser.add_argument("--run_more_rsa_ablations", action="store_true", help="This include RSA only on layers 0-3, 4-7, 8-10, 10-15, [0-3 & 10-15], [4-7 & 8-10], [0-3 & 8-10], [4-7 & 10-15]")
    parser.add_argument("--folder_path_save", type=str, default="../results", help="folder_path")
    parser.add_argument("--use_ddim_inversion", action="store_true",
                        help="If set, use DDIM inversion from a cached trajectory. Otherwise, start from random noise.")
    args = parser.parse_args()

    if args.run_ddim_depth_ablation:
        optimal_depth = run_ddim_depth_ablation(args)
        args.ddim_depth = optimal_depth
    if args.run_rewire_switch_ablation:
        run_rewire_switch_ablation(args)

    run_forward_inference(args)
