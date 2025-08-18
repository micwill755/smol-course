#!/usr/bin/env python3
"""
Simple LoRA Example for Beginners
This shows how LoRA adapts a model with minimal parameters
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

def demonstrate_lora():
    print("🚀 LoRA Beginner Example")
    print("=" * 50)
    
    # Load a small model for demonstration
    model_name = "microsoft/DialoGPT-small"  # Small model for quick demo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📊 Original model parameters: {model.num_parameters():,}")
    
    # Configure LoRA - very simple setup
    lora_config = LoraConfig(
        r=4,                    # Small rank for demo
        lora_alpha=8,          # 2x the rank (common practice)
        lora_dropout=0.1,      # 10% dropout
        bias="none",           # Don't train bias
        target_modules=["c_attn", "c_proj"],  # Target specific attention layers
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to the model
    peft_model = get_peft_model(model, lora_config)
    
    # Show the difference
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    
    print(f"📈 Total parameters: {total_params:,}")
    print(f"🎯 Trainable parameters: {trainable_params:,}")
    print(f"💡 Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    print()
    
    # Test generation BEFORE any training
    print("🔍 BEFORE Training:")
    test_prompt = "Hello, how are you"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 10,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: '{test_prompt}'")
    print(f"Response: '{response}'")
    print()
    
    # Show what LoRA added to the model
    print("🔧 LoRA Modules Added:")
    peft_model.print_trainable_parameters()
    
    print("\n" + "=" * 50)
    print("💡 Key Takeaways:")
    print("• LoRA only trains ~0.5% of parameters")
    print("• Original model weights stay frozen")
    print("• LoRA adds small 'adapter' layers")
    print("• Much faster and cheaper to train")
    print("• Can be easily swapped or removed")

if __name__ == "__main__":
    demonstrate_lora()
