#!/usr/bin/env python3
"""
Fast DialoGPT-small model loader with timing and optimization options
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def load_model_fast(use_quantization=True, use_gpu=True):
    """Load model with timing and optimization options."""
    model_name = "./models/DialoGPT-small"
    
    print(f"🚀 Fast Model Loader")
    print(f"📁 Model path: {model_name}")
    print(f"⚙️  Quantization: {'ON' if use_quantization else 'OFF'}")
    print(f"🖥️  GPU: {'ON' if use_gpu else 'OFF'}")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists(model_name):
        print(f"❌ Model directory not found: {model_name}")
        return None, None
    
    total_start = time.time()
    
    try:
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tok_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        tok_time = time.time() - tok_start
        print(f"✅ Tokenizer loaded in {tok_time:.2f}s")
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with different configurations
        print("📥 Loading model...")
        model_start = time.time()
        
        if use_quantization and use_gpu:
            # Full optimization (slowest initial load, best memory)
            print("🔧 Using 8-bit quantization + GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True
            )
        elif use_gpu and not use_quantization:
            # GPU without quantization (faster load, more memory)
            print("🔧 Using GPU without quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True
            )
        elif not use_gpu:
            # CPU only (fastest load, slowest inference)
            print("🔧 Using CPU only...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                local_files_only=True
            )
        else:
            # Default
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True
            )
        
        model_time = time.time() - model_start
        total_time = time.time() - total_start
        
        print(f"✅ Model loaded in {model_time:.2f}s")
        print(f"⏱️  Total loading time: {total_time:.2f}s")
        print(f"📊 Model parameters: {model.num_parameters():,}")
        
        # Show device info
        if hasattr(model, 'hf_device_map'):
            print(f"🖥️  Device mapping: {model.hf_device_map}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def quick_test(model, tokenizer):
    """Quick model test with timing."""
    if model is None or tokenizer is None:
        return
    
    print("\n🧪 Quick test...")
    test_start = time.time()
    
    inputs = tokenizer("Hello!", return_tensors="pt")
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    test_time = time.time() - test_start
    
    print(f"Input: Hello!")
    print(f"Output: {response}")
    print(f"⚡ Generation time: {test_time:.2f}s")

def main():
    """Main function with options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load DialoGPT-small with different optimization options")
    parser.add_argument("--no-quantization", action="store_true", help="Disable 8-bit quantization (faster load)")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU only (fastest load)")
    parser.add_argument("--no-test", action="store_true", help="Skip model testing")
    
    args = parser.parse_args()
    
    # Configuration
    use_quantization = not args.no_quantization
    use_gpu = not args.cpu_only and torch.cuda.is_available()
    
    if args.cpu_only:
        print("🖥️  CPU-only mode selected (fastest loading)")
    elif args.no_quantization:
        print("⚡ No quantization mode (faster loading, more memory)")
    else:
        print("🔧 Full optimization mode (slower loading, best memory efficiency)")
    
    # Load model
    model, tokenizer = load_model_fast(use_quantization, use_gpu)
    
    # Test if requested
    if model is not None and not args.no_test:
        quick_test(model, tokenizer)
    
    if model is not None:
        print("\n🎉 Model ready for use!")
        
        # Memory usage info
        if torch.cuda.is_available():
            print(f"🔋 GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"🔋 GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    else:
        print("❌ Failed to load model")

if __name__ == "__main__":
    main()
