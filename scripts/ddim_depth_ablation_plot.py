import time
import os
import json
import argparse
import random
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def decode_latent_to_pil(model, latents):
    """Decodes a latent tensor to a PIL Image."""
    x_sample = model.decode_first_stage(latents)
    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()
    return Image.fromarray(x_sample[0].astype(np.uint8))

def generate_and_collect_data(args, model, sampler):
    """
    Runs the articulation process and collects MSE metrics and intermediate images
    for plotting.
    """
    print("Starting data generation and collection...")
    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    set_seed(args.seed)

    # --- Prepare Conditions ---
    prompt = [args.text + args.suffix]
    c = model.get_learned_conditioning(prompt).to(device)
    uc = model.get_learned_conditioning([""]).to(device)
    
    batch_size = args.num_frames
    c_ = {"context": torch.cat([c, c]).repeat(batch_size // 2, 1, 1)}
    uc_ = {"context": uc.repeat(batch_size, 1, 1)}
    
    camera = None
    if args.use_camera:
        camera = get_camera(args.num_frames // 2, elevation=args.camera_elev, azimuth_start=args.camera_azim)
        camera = camera.repeat_interleave(2, dim=0).to(device)
        c_["camera"] = uc_["camera"] = camera
        c_["num_frames"] = uc_["num_frames"] = args.num_frames

    shape = [args.num_frames // 2, args.size // 8, args.size // 8]

    # --- Load Inversion Trajectory ---
    asset_path = f"../results/assets/ddim_inv_trajectories_of_renderings/x_inter_rendered_{args.animal_name}_seed{args.inversion_seed}.torch"
    if not os.path.exists(asset_path):
        raise FileNotFoundError(f"Inversion trajectory not found at {asset_path}. Please run ddim_inversion.py first.")
    
    cached_trajectory = torch.load(asset_path, map_location=device)
    if "rendered" in asset_path:
        cached_trajectory = cached_trajectory[::-1]  # Ensure noise -> clean

    x_T = cached_trajectory[args.ddim_depth].to(device)

    # --- Run Denoising and Collect Intermediates ---
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        samples_ddim, intermediates = sampler.sample(
            S=args.step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=10.0,
            unconditional_conditioning=uc_,
            eta=0.0,
            x_T=x_T
        )

        # --- Process and Collect Data for Plotting ---
        metrics_log = {}
        images_for_plot = {}
        steps_to_capture = [0, 5, 15, 25, 35, 49]  # Timesteps to visualize

        # Get the final reference frame (view 0, frame 0) as the target for MSE
        target_reference_latent = intermediates["pred_x0"][-1][0:1]
        reference_image = decode_latent_to_pil(model, target_reference_latent)
        
        # output_dir = f"{args.folder_path_save}/outputs/{animal_name}_seed{args.seed}_{current_time}/ddim_depth_{ddim_depth}/"
        noise_diff_norm = [float(d.norm(p=2)) for d in [(tn - un) for tn, un in zip(intermediates["model_t"], intermediates["model_uncond"])]]
        # noise_diff_norm = torch.stack(noise_diff_norm, dim=0).cpu().numpy()
        # print("Articulation - Noise Diff Norms", noise_diff_norm.shape)
        # plt.figure(figsize=(10, 6))
        # steps = list(range(len(noise_diff_norm)))
        # plt.plot(steps, noise_diff_norm, 'b-', linewidth=2)
        # plt.xlabel('DDIM Step')
        # plt.ylabel('Noise Diff Norm')
        # plt.title(f'Articulation - DDIM_Depth_{ddim_depth} - Noise Difference Norm vs DDIM Step - {animal_name}')
        # plt.grid(True, alpha=0.3)
        # plt.savefig(f"{output_dir}/noise_diff_norm_recon.png", dpi=150, bbox_inches='tight')
        # plt.close()
        # print("Reconstruction Noise Diff Norm plot saved at:", os.path.join(output_dir, f"ddim_depth_{ddim_depth}_noise_diff_norm_artic.png"))

        mse_values = []
        for t, pred_x0_t in enumerate(intermediates["pred_x0"]):
            # Get the articulated frame (view 0, frame 1) at current step
            articulated_latent_v0f1 = pred_x0_t[1:2]
            
            # Calculate MSE between the decoded images
            mse = F.mse_loss(
                model.decode_first_stage(target_reference_latent),
                model.decode_first_stage(articulated_latent_v0f1)
            ).item()
            mse_values.append(mse)

            if t in steps_to_capture:
                image = decode_latent_to_pil(model, articulated_latent_v0f1)
                images_for_plot[t] = image
                metrics_log[t] = {'mse': mse}
                print(f"Captured Step {t}: MSE = {mse:.4f}")

        final_articulated_latent = samples_ddim[1:2]
        final_articulated_image = decode_latent_to_pil(model, final_articulated_latent)

    return noise_diff_norm, images_for_plot, reference_image, final_articulated_image, metrics_log


def create_aligned_plot(metric_values, images, ref_img, final_img, output_path):
    """Creates and saves the aligned plot."""
    print(f"Creating aligned plot and saving to {output_path}...")
    steps = list(images.keys())
    num_images = len(steps)
    
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, num_images + 2, height_ratios=[1, 2.5])

    # --- Main Plot (Bottom Row) ---
    ax_plot = fig.add_subplot(gs[1, :])
    ax_plot.plot(range(len(metric_values)),metric_values, 'b-', marker='.', label='MSE (Reference vs. Articulated)')
    ax_plot.set_xlabel('DDIM Denoising Step', fontsize=14)
    ax_plot.set_ylabel('Mean Squared Error (MSE)', fontsize=14)
    ax_plot.set_title('Articulation Quality vs. DDIM Step', fontsize=16, pad=20)
    ax_plot.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_plot.legend()
    
    # --- Image Display (Top Row) ---
    image_axes = [fig.add_subplot(gs[0, 0])]
    image_axes[0].imshow(ref_img)
    image_axes[0].set_title("Reference\n(Final Step)", fontsize=12)
    
    for i, step in enumerate(steps):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(images[step])
        ax.set_title(f"Step {step}", fontsize=12)
        image_axes.append(ax)
        
    final_ax = fig.add_subplot(gs[0, num_images + 1])
    final_ax.imshow(final_img)
    final_ax.set_title("Final Articulation\n(Step 50)", fontsize=12)
    image_axes.append(final_ax)

    for ax in image_axes:
        ax.axis('off')

    # --- Draw Connection Lines ---
    for i, step in enumerate(steps):
        xyA = (step,metric_values[step])
        # Position B: middle-bottom of the image axis
        xyB = (0.5, -0.1)
        
        con = ConnectionPatch(xyA=xyA, coordsA=ax_plot.transData,
                              xyB=xyB, coordsB=image_axes[i+1].transAxes,
                              linestyle="--", color="gray", lw=1.5)
        fig.add_artist(con)
        ax_plot.plot(xyA[0], xyA[1], 'ro') # Mark point on curve

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- Model & Path Arguments ---
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--folder_path_save", type=str, default="../results", help="Base folder for outputs")
    parser.add_argument("--animal_name", type=str, default="horse_stallion_highpoly_color_2")

    # --- Generation Arguments ---
    parser.add_argument("--text", type=str, default="a gray horse, dark mane, running, galloping")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--ddim_depth", type=int, default=35)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--inversion_seed", type=int, default=2025)
    
    # --- Camera Arguments ---
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=135)
    
    # --- System Arguments ---
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()

    # --- Setup Environment & Model ---
    output_dir = os.path.join(args.folder_path_save, "aligned_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    model.device = args.device
    model.to(args.device)
    model.eval()
    sampler = DDIMSampler(model)

    # --- Run Main Logic ---
    start_time = time.time()
    metric_data, images_to_plot, ref_image, final_image, metrics_log = generate_and_collect_data(args, model, sampler)
    
    # --- Save Numerical Metrics ---
    metrics_filename = f"metrics_{args.animal_name}_seed{args.seed}_depth{args.ddim_depth}.json"
    metrics_path = os.path.join(output_dir, metrics_filename)
    with open(metrics_path, 'w') as f:
        json.dump({
            "description": "MSE values at specific DDIM steps during articulation.",
            "args": vars(args),
            "metrics_at_captured_steps": metrics_log,
            "all_metric_values": metric_data
        }, f, indent=4)
    print(f"Numerical metrics saved to {metrics_path}")

    # --- Create and Save Plot ---
    plot_filename = f"aligned_plot_{args.animal_name}_seed{args.seed}_depth{args.ddim_depth}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    create_aligned_plot(metric_data, images_to_plot, ref_image, final_image, plot_path)
    
    end_time = time.time()
    print(f"Total script execution time: {end_time - start_time:.2f} seconds.")