# LoRA (Low-Rank Adaptation) Fine-tuning

This directory contains examples and implementations for LoRA fine-tuning, a parameter-efficient method for adapting large language models to specific tasks.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a technique that allows you to fine-tune large language models efficiently by:
- Adding small trainable matrices to existing model layers
- Keeping the original model weights frozen
- Reducing memory usage and training time significantly
- Maintaining model performance while using fewer parameters

## Pre-downloading Models via CLI

To speed up your experiments and avoid waiting for downloads during training, you can pre-download models using the Hugging Face CLI.

### Install Hugging Face Hub

```bash
pip install huggingface_hub
```

### Download Models

#### Option 1: Download to Default Cache
```bash
# Download DialoGPT-small (recommended for tutorials)
huggingface-cli download microsoft/DialoGPT-small

# Download SmolLM2 models (from the smol course)
huggingface-cli download HuggingFaceTB/SmolLM2-135M
huggingface-cli download HuggingFaceTB/SmolLM2-360M
huggingface-cli download HuggingFaceTB/SmolLM2-1.7B

# Download other popular small models
huggingface-cli download distilgpt2
huggingface-cli download gpt2
```

#### Option 2: Download to Local Directory
```bash
# Create a models directory
mkdir -p ./models

# Download to specific local directory
huggingface-cli download microsoft/DialoGPT-small --local-dir ./models/DialoGPT-small
huggingface-cli download HuggingFaceTB/SmolLM2-135M --local-dir ./models/SmolLM2-135M
```

### Enable Faster Downloads

For faster downloads, install and enable `hf_transfer`:

```bash
# Install faster download library
pip install hf_transfer

# Set environment variable for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Then run your downloads
huggingface-cli download microsoft/DialoGPT-small
```

### Using Pre-downloaded Models

Once downloaded, you can load models from local directories:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load from local directory
model_path = "./models/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto"
)
```

### Check Download Progress

Monitor your downloads:

```bash
# Check cache size
du -sh ~/.cache/huggingface/hub/

# Watch cache growth in real-time
watch -n 1 "du -sh ~/.cache/huggingface/hub/"

# List downloaded models
ls ~/.cache/huggingface/hub/
```

## Model Sizes Reference

| Model | Parameters | Approximate Size | Use Case |
|-------|------------|------------------|----------|
| `distilgpt2` | 82M | ~330MB | Quick testing |
| `HuggingFaceTB/SmolLM2-135M` | 135M | ~540MB | Smol course examples |
| `microsoft/DialoGPT-small` | 117M | ~470MB | Conversational AI |
| `HuggingFaceTB/SmolLM2-360M` | 360M | ~1.4GB | Better performance |
| `gpt2` | 124M | ~500MB | Classic baseline |
| `HuggingFaceTB/SmolLM2-1.7B` | 1.7B | ~6.8GB | Production quality |

## Tips for Efficient Downloads

1. **Use `hf_transfer`** - Up to 3x faster downloads
2. **Download overnight** - Large models can take time on slower connections
3. **Use local directories** - Easier to manage and share between projects
4. **Check disk space** - Ensure you have enough space before downloading
5. **Download once, use everywhere** - Models in cache are shared across projects

## Getting Started

1. Pre-download your chosen model using the CLI commands above
2. Run the LoRA fine-tuning examples in this directory
3. Experiment with different model sizes and LoRA configurations

## Next Steps

- Explore the LoRA implementation examples
- Try different rank values and target modules
- Compare training efficiency with full fine-tuning
- Experiment with different base models

Happy fine-tuning! 🚀
