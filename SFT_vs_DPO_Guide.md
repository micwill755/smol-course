# SFT vs DPO: Complete Guide

## Table of Contents
- [Overview](#overview)
- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
- [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
- [Training Pipeline](#training-pipeline)
- [When to Use Each](#when-to-use-each)
- [Code Examples](#code-examples)
- [Comparison Table](#comparison-table)
- [Best Practices](#best-practices)

## Overview

**Fine-tuning** and **post-training** are methods to adapt pre-trained language models for specific tasks and align them with human preferences.

```
Pre-training → Post-training/Fine-tuning → Deployment
     ↓              ↓                        ↓
Base model → Instruction-following → Chat model
(GPT-like)    (SFT, DPO, RLHF)      (ChatGPT-like)
```

## Supervised Fine-Tuning (SFT)

### What is SFT?
Supervised Fine-Tuning teaches a base language model to follow instructions and have conversations using input-output pairs.

### How SFT Works
- **Input**: Instruction or question
- **Output**: Desired response
- **Training**: Model learns to map inputs to outputs

### Example Transformation

**Before SFT (Base model):**
```
Input: "What is 2+2?"
Output: "The number 2 is a mathematical concept that represents..."
```

**After SFT (Instruction-tuned model):**
```
Input: "What is 2+2?"
Output: "2+2 equals 4."
```

### SFT Data Format
```python
# Chat format
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]

# Or simple text format
text = "Human: What is the capital of France?\n\nAssistant: The capital of France is Paris."
```

### SFT Code Example
```python
from trl import SFTTrainer
from transformers import TrainingArguments

# Training arguments
training_args = TrainingArguments(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    bf16=False,
    fp16=False,
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=conversation_dataset,
    processing_class=tokenizer,
)

# Train
trainer.train()
```

## Direct Preference Optimization (DPO)

### What is DPO?
DPO aligns models with human preferences by training on pairs of preferred vs rejected responses, without requiring a separate reward model.

### How DPO Works
- **Input**: Same prompt
- **Preferred**: Better response
- **Rejected**: Worse response
- **Training**: Model learns to prefer better responses

### DPO vs RLHF
| Aspect | DPO | RLHF |
|--------|-----|------|
| **Complexity** | Simple | Complex |
| **Reward Model** | Not needed | Required |
| **Training Steps** | 1 step | 3 steps |
| **Stability** | More stable | Can be unstable |

### DPO Data Format
```python
# Preference pairs
{
    "prompt": "Explain quantum physics",
    "chosen": "Quantum physics studies matter and energy at the smallest scales...",
    "rejected": "Quantum physics is like, really complicated stuff with particles and waves and stuff..."
}
```

### DPO Code Example
```python
from trl import DPOTrainer
from transformers import TrainingArguments

# Training arguments
training_args = TrainingArguments(
    output_dir="./dpo_output",
    max_steps=500,
    per_device_train_batch_size=2,
    learning_rate=1e-6,  # Lower learning rate for DPO
    bf16=False,
    fp16=False,
)

# DPO Trainer
trainer = DPOTrainer(
    model=sft_model,  # Usually start with SFT model
    args=training_args,
    train_dataset=preference_dataset,
    processing_class=tokenizer,
)

# Train
trainer.train()
```

## Training Pipeline

### Standard Pipeline (Recommended)
```python
# Step 1: Base Model
base_model = "HuggingFaceTB/SmolLM2-1.7B"

# Step 2: SFT - Teach instruction following
sft_trainer = SFTTrainer(model=base_model, train_dataset=instruction_dataset)
sft_model = sft_trainer.train()

# Step 3: DPO - Align with preferences
dpo_trainer = DPOTrainer(model=sft_model, train_dataset=preference_dataset)
final_model = dpo_trainer.train()
```

### Alternative: Skip SFT (Sometimes Possible)
```python
# Option 1: Use pre-trained instruct model
instruct_model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # Already has SFT
dpo_trainer = DPOTrainer(model=instruct_model, train_dataset=preference_dataset)

# Option 2: Direct DPO (risky)
base_model = "HuggingFaceTB/SmolLM2-1.7B"
dpo_trainer = DPOTrainer(model=base_model, train_dataset=preference_dataset)
```

## When to Use Each

### Use SFT When:
- ✅ Starting with a base model (not instruction-tuned)
- ✅ Model doesn't follow instructions well
- ✅ You have instruction-response datasets
- ✅ Teaching basic conversational abilities
- ✅ First time fine-tuning (learning purposes)

### Use DPO When:
- ✅ Model already follows instructions (post-SFT or pre-trained instruct)
- ✅ You want to align with specific preferences
- ✅ You have preference pair datasets
- ✅ Refining response quality and style
- ✅ Avoiding harmful or unwanted outputs

### Skip SFT When:
- ✅ Using pre-trained instruct models (SmolLM2-Instruct, DialoGPT)
- ✅ Model already follows instructions adequately
- ✅ Only need preference alignment
- ✅ Domain-specific models that already work well

## Code Examples

### Complete SFT Setup
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# Load model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-1.7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set chat template
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "Human: {{ message['content'] }}\n\n"
    "{% elif message['role'] == 'assistant' %}"
    "Assistant: {{ message['content'] }}\n\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "Assistant: "
    "{% endif %}"
)

# Load dataset
ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")

# Training arguments
training_args = TrainingArguments(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    bf16=False,
    fp16=False,
    dataloader_num_workers=0,
    report_to="none",
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    processing_class=tokenizer,
)

# Train
trainer.train()

# Save
trainer.save_model("./my-sft-model")
tokenizer.save_pretrained("./my-sft-model")
```

### Complete DPO Setup
```python
from trl import DPOTrainer
from datasets import load_dataset

# Load SFT model (or pre-trained instruct model)
sft_model = AutoModelForCausalLM.from_pretrained("./my-sft-model")
tokenizer = AutoTokenizer.from_pretrained("./my-sft-model")

# Load preference dataset
preference_ds = load_dataset("Anthropic/hh-rlhf")

# DPO training arguments (lower learning rate)
dpo_args = TrainingArguments(
    output_dir="./dpo_output",
    max_steps=500,
    per_device_train_batch_size=2,
    learning_rate=1e-6,  # Much lower than SFT
    logging_steps=10,
    save_steps=50,
    bf16=False,
    fp16=False,
    dataloader_num_workers=0,
    report_to="none",
)

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model=sft_model,
    args=dpo_args,
    train_dataset=preference_ds["train"],
    processing_class=tokenizer,
)

# Train
dpo_trainer.train()

# Save
dpo_trainer.save_model("./my-dpo-model")
```

## Comparison Table

| Aspect | SFT | DPO |
|--------|-----|-----|
| **Purpose** | Teach instruction following | Align with preferences |
| **Data Type** | Instruction-response pairs | Preference pairs (chosen/rejected) |
| **Training Objective** | Maximize likelihood of responses | Optimize preference ranking |
| **Learning Rate** | Higher (5e-5) | Lower (1e-6) |
| **When to Use** | Base models, first fine-tuning | After SFT, preference alignment |
| **Data Format** | `{"messages": [...]}` | `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| **Complexity** | Simple | Moderate |
| **Hardware Requirements** | Standard | Similar to SFT |
| **Training Time** | Longer (more steps) | Shorter (fewer steps) |

## Best Practices

### SFT Best Practices
```python
# 1. Set appropriate chat template
tokenizer.chat_template = "..."

# 2. Use reasonable learning rates
learning_rate=5e-5  # Good starting point

# 3. Monitor training loss
logging_steps=10

# 4. Save regularly
save_steps=100

# 5. Use appropriate batch sizes for your hardware
per_device_train_batch_size=4  # Adjust based on GPU memory
```

### DPO Best Practices
```python
# 1. Start with SFT model (usually better)
model = load_sft_model()

# 2. Use lower learning rates
learning_rate=1e-6  # Much lower than SFT

# 3. Smaller batch sizes (DPO is memory intensive)
per_device_train_batch_size=2

# 4. Quality preference data is crucial
# Ensure clear preference differences

# 5. Monitor preference accuracy
# Track how often model chooses preferred responses
```

### Hardware Considerations
```python
# For Apple Silicon (MPS) or CPU
training_args = TrainingArguments(
    bf16=False,  # MPS doesn't support
    fp16=False,  # MPS doesn't support
    dataloader_num_workers=0,  # Required for MPS
    per_device_train_batch_size=2,  # Smaller for stability
)

# For modern GPUs (RTX 30/40, A100)
training_args = TrainingArguments(
    bf16=True,  # Faster and stable
    per_device_train_batch_size=8,  # Larger batches
    dataloader_num_workers=4,
)
```

## Troubleshooting

### Common SFT Issues
```python
# Issue: Chat template not set
# Solution:
tokenizer.chat_template = "..."

# Issue: Out of memory
# Solution:
per_device_train_batch_size=2  # Reduce batch size
gradient_accumulation_steps=2  # Maintain effective batch size

# Issue: Model not following instructions
# Solution:
max_steps=2000  # Train longer
learning_rate=1e-4  # Try higher learning rate
```

### Common DPO Issues
```python
# Issue: Training unstable
# Solution:
learning_rate=5e-7  # Even lower learning rate
per_device_train_batch_size=1  # Smaller batches

# Issue: No improvement
# Solution:
# Check preference data quality
# Ensure clear preference differences
# Start with better SFT model

# Issue: Model becomes worse
# Solution:
# Use lower learning rate
# Train for fewer steps
# Check data quality
```

## Conclusion

- **SFT**: Essential for teaching instruction following to base models
- **DPO**: Powerful for aligning with human preferences
- **Pipeline**: SFT → DPO usually works best
- **Shortcuts**: Can skip SFT if using pre-trained instruct models
- **Learning**: Try both to understand the differences

For the smol-course, complete SFT first, then experiment with DPO to see how each method transforms model behavior!
