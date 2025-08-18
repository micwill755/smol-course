# LoRA Examples for Beginners

This folder contains simple examples to help understand LoRA (Low-Rank Adaptation) fine-tuning.

## 📁 Files

### `lora_example.py`
- **Purpose**: Basic demonstration of how LoRA works
- **Shows**: Parameter reduction, model structure changes
- **Runtime**: ~30 seconds
- **Good for**: Understanding LoRA concepts

### `lora_training_example.py`
- **Purpose**: Complete training example with before/after comparison
- **Shows**: Actual fine-tuning process, response changes
- **Runtime**: ~2-3 minutes
- **Good for**: Seeing LoRA in action

## 🚀 How to Run

```bash
# Navigate to examples folder
cd lora_examples

# Run basic example
python3 lora_example.py

# Run training example
python3 lora_training_example.py
```

## 🎯 Key Concepts Demonstrated

- **Parameter Efficiency**: Only ~0.3% of parameters are trainable
- **Memory Savings**: Much less GPU memory required
- **Speed**: Faster training compared to full fine-tuning
- **Flexibility**: Easy to add/remove adaptations

## 🔗 Connection to Main Project

These examples use the same LoRA configuration principles as the main multi-GPU training:

```python
# From main project
peft_config = LoraConfig(
    r=6,                    # Same rank dimension
    lora_alpha=8,          # Same scaling factor
    lora_dropout=0.05,     # Same dropout rate
    target_modules="all-linear",  # More comprehensive targeting
    task_type="CAUSAL_LM"
)
```

The main difference is that the multi-GPU setup targets `"all-linear"` modules for more comprehensive adaptation.
