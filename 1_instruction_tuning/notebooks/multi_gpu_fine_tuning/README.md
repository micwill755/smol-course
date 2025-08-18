# Multi-GPU Fine-Tuning Setup

This folder contains everything needed to fine-tune SmolLM2 on 4 GPUs using Distributed Data Parallel (DDP).

## 🚀 Quick Start

```bash
# Navigate to this directory
cd multi_gpu_fine_tuning

# Set your Hugging Face token (if needed)
export HF_TOKEN="your_token_here"

# Launch training on all 4 GPUs
./launch_multigpu.sh
```

## 📁 Files Overview

- **`sft_multigpu_training.py`** - Main training script optimized for 4 GPUs
- **`launch_multigpu.sh`** - Easy launch script using torchrun
- **`accelerate_config.yaml`** - Alternative accelerate configuration
- **`requirements_multigpu.txt`** - Required Python packages
- **`README_MultiGPU.md`** - Detailed documentation and troubleshooting

## ⚡ Performance

With 4 NVIDIA A10G GPUs:
- **~4x faster training** compared to single GPU
- **Effective batch size of 8** (2 per GPU × 4 GPUs)
- **1000 steps complete in ~10-15 minutes**
- **Automatic gradient synchronization** across GPUs

## 🔧 Key Features

- **Tested and working** on 4 GPU setup
- **Proper DDP configuration** with NCCL backend
- **bfloat16 mixed precision** for optimal performance
- **Automatic model distribution** across GPUs
- **TensorBoard logging** for monitoring
- **Hugging Face Hub integration** for model sharing

## 📊 Expected Output

The training will:
1. Load model across all 4 GPUs
2. Test generation before training
3. Train for 1000 steps with distributed batching
4. Save the fine-tuned model
5. Test generation after training
6. Push to Hugging Face Hub

For detailed instructions and troubleshooting, see `README_MultiGPU.md`.
