import os
import re
import json

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageDraw
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


def natural_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", name)]


class PLJ(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):
        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = configs.num_frames
        self.frame_interval = configs.frame_interval
        self.condition_root = getattr(configs, "condition_root", None)
        self.use_condition = bool(getattr(configs, "use_condition", False))
        self.data_all = self.load_video_frames(self.data_path)

    def __getitem__(self, index):
        vframes = self.data_all[index]
        total_frames = len(vframes)

        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        frame_indices = np.linspace(
            start_frame_ind,
            end_frame_ind - 1,
            num=self.target_video_len,
            dtype=int,
        )
        select_video_frames = [vframes[i] for i in frame_indices]

        video_frames = []
        for path in select_video_frames:
            frame = self.load_rgb_frame(path)
            frame = torch.as_tensor(np.array(frame, dtype=np.uint8, copy=True)).unsqueeze(0)
            video_frames.append(frame)

        video_clip = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)
        video_clip = self.transform(video_clip)
        item = {"video": video_clip, "video_name": 1}
        if self.use_condition:
            item["condition"] = self.render_building_condition(select_video_frames)
        return item

    def __len__(self):
        return len(self.data_all)

    def load_video_frames(self, dataroot):
        data_all = []
        min_frames = self.target_video_len * self.frame_interval
        for root, _, files in os.walk(dataroot):
            frames = [
                os.path.join(root, item)
                for item in files
                if is_image_file(item)
            ]
            if len(frames) >= min_frames:
                data_all.append(sorted(frames, key=natural_key))
        return sorted(data_all, key=lambda frames: frames[0])

    @staticmethod
    def load_rgb_frame(path):
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as exc:
            print(f"Warning: failed to load image {path}: {exc}. Using a black frame.")
            return Image.new("RGB", (256, 256), color="black")

    @staticmethod
    def height_index_from_frame_name(path, fallback):
        stem = os.path.splitext(os.path.basename(path))[0]
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else fallback

    def render_building_condition(self, frame_paths):
        if self.condition_root is None:
            raise ValueError("PLJ condition is enabled, but condition_root is not set.")

        clip_name = os.path.basename(os.path.dirname(frame_paths[0]))
        condition_id = clip_name.split("_", 1)[0]
        condition_path = os.path.join(self.condition_root, f"{condition_id}.json")
        with open(condition_path, "r") as f:
            buildings = json.load(f)

        frames = []
        for i, frame_path in enumerate(frame_paths):
            height_idx = self.height_index_from_frame_name(frame_path, i + 1)
            img = Image.new("L", (self.configs.image_size, self.configs.image_size), 0)
            draw = ImageDraw.Draw(img)
            for polygon, height_values in buildings:
                height = float(height_values[0])
                if height >= height_idx:
                    draw.polygon([tuple(point) for point in polygon], fill=255)
            frame = torch.as_tensor(np.array(img, dtype=np.uint8, copy=True)).unsqueeze(0).unsqueeze(0)
            frames.append(frame)

        condition = torch.cat(frames, dim=0).float() / 255.0
        return condition
