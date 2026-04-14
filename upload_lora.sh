#!/bin/bash
# 上传两个 LoRA 权重到 HuggingFace
# 前提：已经 hf auth login
set -e

HF=/home/work/conda/envs/trainer/bin/hf

cd /home/work/tmp/trainer

echo "[1/2] Uploading no_iou LoRA..."
$HF upload \
    qwen-edit-lora-no-iou \
    lora_saves_no_iou/checkpoint-200/pytorch_lora_weights.safetensors \
    pytorch_lora_weights.safetensors \
    --repo-type=model

echo ""
echo "[2/2] Uploading with_iou LoRA..."
$HF upload \
    qwen-edit-lora-with-iou \
    lora_saves_with_iou/checkpoint-200/pytorch_lora_weights.safetensors \
    pytorch_lora_weights.safetensors \
    --repo-type=model

echo ""
echo "Done."
