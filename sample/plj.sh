#!/bin/sh
python sample/sample_plj.py \
  --config ./configs/plj/plj_train.yaml \
  --ckpt "$1" \
  --save_video_path ./sample_videos/plj \
  --sample_index "${2:-0}"
