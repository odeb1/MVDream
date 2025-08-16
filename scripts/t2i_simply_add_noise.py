import os
import sys
import random
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

FORMAT_INTERLEAVED = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1,
        ddim_inversion=False, start_time_step=35):
    if type(prompt)!=list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        prompt = "cow running legs bent"
        c0 = model.get_learned_conditioning(prompt).to(device)
        c1 = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": torch.cat([c0, c1]).repeat(batch_size//2,1,1)}
        ##### for vf, context in enumerate(c_['context']):  print(vf, context.shape, context[4:16])
        # uc = model.get_learned_conditioning("resting, wings tucked in, wings folded tightly against its sides").to(device)
        uc = model.get_learned_conditioning("resting legs straight").to(device)
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames
        shape = [4, image_size // 8, image_size // 8] # [4, 32, 32]

        x = []
        for i in range(4):
            # Load the image
            x.append(to_tensor(Image.open(os.path.join('/data/results/outputs/humming_8_views_seed2222_250805170217/ddim_depth_25/', f'sample_{i}.png'))))
        x = torch.stack(x).to(device) * 2.0 - 1.0
        x = F.interpolate(x, (256, 256))
        print(x.shape)
        x_T = model.encode_first_stage(x).mean #.sample() # DiagonalGaussianDistribution
        x_T = x_T * 0.18215

        clean_latents = torch.split(x_T, 4)
        for v, z in enumerate(clean_latents):
            x_sample = model.decode_first_stage(z)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
            x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
            # Image.fromarray(x_sample).save(f"ddim_inv_artefacts/clean_latents_view{v}.png")

        if FORMAT_INTERLEAVED:
            x_T = x_T.repeat_interleave(2, dim=0)    # AABBCCDD
        else:
            repeater = num_frames // 4
            x_T = x_T.repeat(repeater, 1, 1, 1)      # ABCDABCD

        intermediates = [x_T]
        sampler.make_schedule(ddim_num_steps=args.step, ddim_eta=0, verbose=True)
        # sampler.sqrt_alphas_cumprod: (1000,)
        # noise = torch.rand_like(x_T).cuda()
        for i in range(args.step):
            i = torch.tensor([i]).cuda()
            z = sampler.stochastic_encode(x_T, i, use_original_steps=False, noise=None)
            intermediates.append(z)
        
        for t, x_t in enumerate(intermediates):
            x_sample = model.decode_first_stage(x_t)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
            x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
            # Image.fromarray(x_sample).save(f"ddim_inv_artefacts/x_inter_t={t}.png")

        samples_ddim, intermediates = sampler.sample(S=step, conditioning=c_,
                                    batch_size=batch_size, shape=shape,
                                    verbose=False, 
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc_,
                                    eta=ddim_eta, x_T=intermediates[-args.start_time_step+5],
                                    start_time_step=start_time_step,)
        for t, x_t in enumerate(intermediates['x_inter']):
            x_sample = model.decode_first_stage(x_t)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
            x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
            # Image.fromarray(x_sample).save(f"forward_cache_artefacts/x_inter_t={t+args.start_time_step}.png")
    
        sys.exit(0)

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="a cow standing still legs straight")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--start_time_step", type=int, default=35, help="DDIM inversion start time step")
    parser.add_argument("--ddim_inversion", action="store_true")
    parser.add_argument("--num_frames", type=int, default=8, help="num of frames (views) to generate")
    parser.add_argument("--num_rows", type=int, default=1, help="number of rows to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
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
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
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
    images = []
    for j in range(args.num_rows):
        img = t2i(model, args.size, t, uc, sampler, step=args.step, scale=10, batch_size=batch_size, ddim_eta=0.0, 
                dtype=dtype, device=device, camera=camera, num_frames=args.num_frames, ddim_inversion=args.ddim_inversion, start_time_step=args.start_time_step)
        for i, im in enumerate(img):
            Image.fromarray(im).save(f"sample_{i}.png")
        img = np.concatenate(img, 1)
        images.append(img)
    images = np.concatenate(images, 0)
    Image.fromarray(images).save(f"sample.png")