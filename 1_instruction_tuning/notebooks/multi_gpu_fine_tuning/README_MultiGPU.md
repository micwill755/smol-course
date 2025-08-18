# Multi-GPU SFT Training Setup

This setup allows you to train the SmolLM2 model on all 4 GPUs using Distributed Data Parallel (DDP).

## Files Created

- `sft_multigpu_training.py` - Main training script optimized for multi-GPU
- `launch_multigpu.sh` - Launch script using torchrun
- `accelerate_config.yaml` - Configuration for accelerate launcher
- `requirements_multigpu.txt` - Required packages

## Key Changes for Multi-GPU Training

1. **Model Loading**: Removed `.to(device)` to let DDP handle device placement
2. **Configuration**: Optimized SFTConfig for multi-GPU performance
3. **Batch Size**: Set `per_device_train_batch_size=2` (effective batch size = 8 with 4 GPUs)
4. **Mixed Precision**: Enabled `bf16=True` for better performance
5. **DDP Settings**: Configured NCCL backend and other DDP optimizations

## How to Run

### Option 1: Using torchrun (Recommended)

```bash
# Navigate to the notebook directory
cd /home/ubuntu/smol/1_instruction_tuning/notebooks

# Set your Hugging Face token (if not already set)
export HF_TOKEN="your_token_here"

# Run the training
./launch_multigpu.sh
```

### Option 2: Using accelerate

```bash
# First time setup (run once)
accelerate config --config_file accelerate_config.yaml

# Run training
accelerate launch --config_file accelerate_config.yaml sft_multigpu_training.py
```

### Option 3: Direct torchrun command

```bash
torchrun --nproc_per_node=4 --master_port=29500 sft_multigpu_training.py
```

## Performance Expectations

With 4 GPUs, you should see:
- **4x faster training** compared to single GPU
- **Effective batch size of 8** (2 per GPU × 4 GPUs)
- **Better GPU utilization** with proper data loading
- **Automatic gradient synchronization** across GPUs

## Monitoring

- **TensorBoard logs**: Check `./sft_output_multigpu/runs/`
- **GPU usage**: Run `nvidia-smi` in another terminal
- **Training progress**: Watch the console output

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `per_device_train_batch_size` from 2 to 1
2. **NCCL timeout**: Check network connectivity between GPUs
3. **Port already in use**: Change `--master_port` to a different number

### Checking GPU Usage

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Environment Variables

```bash
# For debugging NCCL issues
export NCCL_DEBUG=INFO

# For better performance
export CUDA_LAUNCH_BLOCKING=0
export NCCL_TREE_THRESHOLD=0
```

## Expected Output

The script will:
1. Load the model and dataset
2. Test generation before training
3. Train for 1000 steps across 4 GPUs
4. Save the model
5. Test generation after training
6. Push to Hugging Face Hub

Training should complete much faster than single-GPU training while maintaining the same quality.
