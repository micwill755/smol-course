#!/bin/bash

# Multi-GPU Training Launch Script - WORKING VERSION
# This script launches the SFT training on all 4 GPUs

echo "🚀 Starting multi-GPU training on 4 GPUs..."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"

# Set environment variables for optimal performance
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN environment variable not set"
    echo "   Set it with: export HF_TOKEN='your_token_here'"
fi

# Launch with torchrun
echo "🔥 Launching with torchrun..."
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    sft_multigpu_training.py

echo "✅ Training completed!"
