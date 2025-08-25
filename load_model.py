#!/usr/bin/env python3
"""
Load DialoGPT-small model from local directory
This script demonstrates how to load a pre-downloaded model efficiently.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def check_model_files(model_path):
    """Check if all required model files exist."""
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ]
    
    # Check for model weights (either safetensors or pytorch_model.bin)
    model_files = ["model.safetensors", "pytorch_model.bin"]
    has_model_weights = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if not has_model_weights:
        missing_files.append("model weights (model.safetensors or pytorch_model.bin)")
    
    return missing_files

def load_model_local():
    """Load model from local directory."""
    # Model configuration
    model_name = "./models/DialoGPT-small"  # Local path
    
    print(f"🔍 Loading model from: {model_name}")
    print(f"📁 Absolute path: {os.path.abspath(model_name)}")
    
    # Check if model directory exists
    if not os.path.exists(model_name):
        print(f"❌ Error: Model directory '{model_name}' does not exist!")
        print("💡 Tip: Download the model first using:")
        print("   huggingface-cli download microsoft/DialoGPT-small --local-dir ./models/DialoGPT-small")
        return None, None
    
    # Check if all required files exist
    missing_files = check_model_files(model_name)
    if missing_files:
        print(f"❌ Error: Missing required files: {', '.join(missing_files)}")
        print("💡 Tip: Re-download the model to ensure all files are present")
        return None, None
    
    try:
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("✅ Tokenizer loaded successfully!")
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("🔧 Added padding token")
        
        # Load model with 8-bit quantization to save memory
        print("📥 Loading model (this may take a moment)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True  # Ensure we only use local files
        )
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model parameters: {model.num_parameters():,}")
        print(f"💾 Model size: ~{model.num_parameters()/1e6:.1f}M parameters")
        
        # Display model info
        print(f"🏗️  Model architecture: {model.config.model_type}")
        if hasattr(model.config, 'n_layer'):
            print(f"📚 Number of layers: {model.config.n_layer}")
        if hasattr(model.config, 'n_head'):
            print(f"🧠 Attention heads: {model.config.n_head}")
        if hasattr(model.config, 'n_embd'):
            print(f"🔢 Hidden size: {model.config.n_embd}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Try downloading the model again or check the file permissions")
        return None, None

def load_model_remote():
    """Load model from Hugging Face Hub as fallback."""
    model_name = "microsoft/DialoGPT-small"
    
    print(f"🌐 Downloading model from Hugging Face Hub: {model_name}")
    print("⏳ This may take a few minutes on first download...")
    
    try:
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded!")
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("🔧 Added padding token")
        
        # Load model
        print("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model parameters: {model.num_parameters():,}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model from Hub: {e}")
        return None, None

def test_model(model, tokenizer):
    """Test the loaded model with a simple generation."""
    if model is None or tokenizer is None:
        print("❌ Cannot test model - model or tokenizer is None")
        return
    
    print("\n🧪 Testing model with sample generation...")
    
    # Test input
    test_input = "Hello, how are you?"
    print(f"Input: {test_input}")
    
    try:
        # Tokenize input
        inputs = tokenizer.encode(test_input, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 20,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {response}")
        print("✅ Model test successful!")
        
    except Exception as e:
        print(f"❌ Error during model test: {e}")

def main():
    """Main function."""
    print("🚀 DialoGPT-small Model Loader")
    print("=" * 50)
    
    # Try loading from local directory first
    model, tokenizer = load_model_local()
    
    # If local loading fails, try remote
    if model is None:
        print("\n🔄 Falling back to remote download...")
        model, tokenizer = load_model_remote()
    
    # Test the model if successfully loaded
    if model is not None:
        test_model(model, tokenizer)
        print("\n🎉 All done! Model is ready for use.")
    else:
        print("\n❌ Failed to load model. Please check your setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()
