import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets import get_dataset
from diffusion import create_diffusion
from models import get_models
from train import load_vae, save_training_preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plj/plj_train.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="manual_previews")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for preview sampling")

    cfg = OmegaConf.load(args.config)
    cfg.latent_size = cfg.image_size // 8
    cfg.vis_sampling_steps = args.steps

    model = get_models(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    key = "ema" if args.ema and "ema" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.eval()

    vae = load_vae(cfg.pretrained_model_path, device)
    vae.eval()
    diffusion = create_diffusion("")

    dataset = get_dataset(cfg)
    batch = dataset[args.index]
    batch = {
        "video": batch["video"].unsqueeze(0),
        "condition": batch["condition"].unsqueeze(0),
    }
    os.makedirs(args.out, exist_ok=True)
    save_training_preview(cfg, model, vae, diffusion, batch, int(ckpt.get("step", 0)), args.out, device)
    print(f"Saved preview to {args.out}")


if __name__ == "__main__":
    main()
