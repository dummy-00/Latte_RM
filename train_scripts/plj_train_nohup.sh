#!/bin/sh
mkdir -p logs
LOG_FILE="logs/plj_train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting PLJ Latte training with nohup"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}"
echo "NPROC=${NPROC:-2}"
echo "Log: ${LOG_FILE}"

nohup sh train_scripts/plj_train.sh > "${LOG_FILE}" 2>&1 &
echo "PID: $!"
