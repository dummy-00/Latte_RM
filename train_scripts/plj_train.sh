#!/bin/sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
NPROC=${NPROC:-2}

python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch =", torch.__version__, "torch.version.cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available(), "device_count =", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment. Check driver, CUDA_VISIBLE_DEVICES, and PyTorch CUDA wheel.")
PY

torchrun --nnodes=1 --nproc_per_node="${NPROC}" --master_port=29519 train.py --config ./configs/plj/plj_train.yaml
