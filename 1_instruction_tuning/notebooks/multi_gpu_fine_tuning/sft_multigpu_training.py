#!/usr/bin/env python3
"""
Multi-GPU Supervised Fine-Tuning Script - WORKING VERSION
Optimized for training on 4 GPUs using DistributedDataParallel
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, setup_chat_format
from huggingface_hub import login

def main():
    # Login to Hugging Face (make sure HF_TOKEN is set as environment variable)
    # Uncomment the line below if you need interactive login
    # login()
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Load the model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M"
    print(f"Loading model: {model_name}")
    
    # Don't move to device - let DDP handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
        device_map=None  # Let DDP handle device mapping
    )
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    
    # Set up the chat format
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
    
    # Set our name for the finetune
    finetune_name = "SmolLM2-FT-MyDataset-MultiGPU"
    finetune_tags = ["smol-course", "module_1", "multi-gpu"]
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")
    print(f"Dataset loaded - Train: {len(ds['train'])}, Test: {len(ds['test'])}")
    
    # Configure SFTTrainer for multi-GPU training
    sft_config = SFTConfig(
        output_dir="./sft_output_multigpu",
        max_steps=1000,
        per_device_train_batch_size=2,  # With 4 GPUs, effective batch size = 8
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        eval_strategy="steps",  # Correct parameter name
        
        # Multi-GPU optimizations
        bf16=True,  # Use bfloat16 for better performance on modern GPUs
        fp16=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        gradient_accumulation_steps=1,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",  # Best backend for multi-GPU CUDA
        remove_unused_columns=True,  # Let SFTTrainer handle column removal
        
        # Performance optimizations
        group_by_length=True,
        warmup_steps=100,
        weight_decay=0.01,
        
        # Model saving optimizations
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Reporting
        report_to=["tensorboard"],  # Log to tensorboard
        run_name=f"{finetune_name}-{torch.cuda.device_count()}gpu",
        hub_model_id=finetune_name,
    )
    
    # Initialize the SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        # Don't specify data_collator - let SFTTrainer handle it automatically
        # The dataset has 'messages' field which SFTTrainer will use automatically
    )
    
    # Test generation before training (only on main process)
    if trainer.is_world_process_zero():
        print("\n" + "="*50)
        print("TESTING MODEL BEFORE TRAINING")
        print("="*50)
        test_generation(model, tokenizer)
    
    # Train the model
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    trainer.train()
    
    # Save the model (only on main process)
    if trainer.is_world_process_zero():
        print("Saving model...")
        trainer.save_model(f"./{finetune_name}")
        
        # Test generation after training
        print("\n" + "="*50)
        print("TESTING MODEL AFTER TRAINING")
        print("="*50)
        test_generation(model, tokenizer)
        
        # Push to hub
        print("Pushing to Hugging Face Hub...")
        trainer.push_to_hub(tags=finetune_tags)
        print("Training completed successfully!")

def test_generation(model, tokenizer):
    """Test model generation capabilities"""
    device = next(model.parameters()).device
    
    prompts = [
        "Write a haiku about programming",
        "Explain what machine learning is",
        "How do I cook pasta?"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        # Format with chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        print("-" * 40)

if __name__ == "__main__":
    main()
