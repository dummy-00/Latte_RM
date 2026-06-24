from huggingface_hub import snapshot_download


if __name__ == "__main__":
    path = snapshot_download(
        repo_id="stabilityai/sd-vae-ft-ema",
        local_dir="/home/plj/Latte/pretrained/sd-vae-ft-ema",
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(path)
