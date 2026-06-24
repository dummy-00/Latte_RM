import argparse
import os
import sys

try:
    import utils
    from diffusion import create_diffusion
    from utils import find_model
except Exception:
    sys.path.append(os.path.split(sys.path[0])[0])
    import utils
    from diffusion import create_diffusion
    from utils import find_model

import imageio
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from einops import rearrange
from models import get_models
from omegaconf import OmegaConf


def load_vae(pretrained_model_path, device):
    if os.path.isdir(pretrained_model_path):
        if os.path.exists(os.path.join(pretrained_model_path, "vae", "config.json")):
            return AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device)
        return AutoencoderKL.from_pretrained(pretrained_model_path).to(device)
    return AutoencoderKL.from_pretrained(pretrained_model_path).to(device)


def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("PLJ sampling requires a CUDA GPU.")

    args.latent_size = args.image_size // 8
    model = get_models(args).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict["ema"] if "ema" in state_dict else state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = load_vae(args.pretrained_model_path, device)

    from datasets import get_dataset

    dataset = get_dataset(args)
    item = dataset[args.sample_index]
    condition = item["condition"].unsqueeze(0).to(device)
    condition = F.interpolate(
        condition.flatten(0, 1),
        size=(args.latent_size, args.latent_size),
        mode="nearest",
    )
    condition = condition.unflatten(0, (1, args.num_frames)).contiguous()

    z = torch.randn(1, args.num_frames, 4, args.latent_size, args.latent_size, device=device)
    model_kwargs = dict(condition=condition, use_fp16=False)

    if args.sample_method == "ddim":
        samples = diffusion.ddim_sample_loop(
            model,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )
    else:
        samples = diffusion.p_sample_loop(
            model,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )

    b, f, c, h, w = samples.shape
    samples = rearrange(samples, "b f c h w -> (b f) c h w")
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, "(b f) c h w -> b f c h w", b=b)

    os.makedirs(args.save_video_path, exist_ok=True)
    video = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255)
    video = video.to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    save_path = os.path.join(args.save_video_path, "sample_plj.mp4")
    imageio.mimwrite(save_path, video, fps=8, quality=9)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/plj/plj_train.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/plj")
    parser.add_argument("--sample_index", type=int, default=0)
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    omega_conf.sample_index = args.sample_index
    main(omega_conf)
