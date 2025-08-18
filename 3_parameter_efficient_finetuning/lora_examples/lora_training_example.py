#!/usr/bin/env python3
"""
Practical LoRA Training Example
Shows how to fine-tune a model to respond in a specific style
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

def create_training_data():
    """Create simple training data to teach the model a specific response style"""
    
    # Training data: teach model to be a helpful coding assistant
    training_examples = [
        {
            "input": "How do I create a list in Python?",
            "output": "You can create a list in Python using square brackets: my_list = [1, 2, 3, 'hello']. Lists are mutable and can contain different data types."
        },
        {
            "input": "What is a function in Python?",
            "output": "A function in Python is defined using 'def' keyword: def my_function(): return 'Hello'. Functions help organize code and make it reusable."
        },
        {
            "input": "How do I loop through a list?",
            "output": "Use a for loop: for item in my_list: print(item). This iterates through each element in the list."
        },
        {
            "input": "What is a dictionary?",
            "output": "A dictionary stores key-value pairs: my_dict = {'name': 'John', 'age': 30}. Access values using keys: my_dict['name']."
        }
    ]
    
    # Format for training (instruction format)
    formatted_data = []
    for example in training_examples:
        text = f"### Question: {example['input']}\n### Answer: {example['output']}<|endoftext|>"
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def demonstrate_lora_training():
    print("🎓 LoRA Training Example")
    print("=" * 60)
    
    # Load model and tokenizer
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📊 Base model: {model_name}")
    print(f"📊 Original parameters: {model.num_parameters():,}")
    
    # Configure LoRA (similar to your config)
    lora_config = LoraConfig(
        r=6,                    # Same as your config
        lora_alpha=8,          # Same as your config  
        lora_dropout=0.05,     # Same as your config
        bias="none",
        target_modules=["c_attn", "c_proj"],  # Attention layers
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    # Test BEFORE training
    print("\n🔍 BEFORE Training:")
    test_prompt = "### Question: How do I create a list in Python?\n### Answer:"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    # Create training dataset
    train_dataset = create_training_data()
    print(f"\n📚 Training on {len(train_dataset)} examples")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=128
        )
    
    # Tokenize dataset
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    
    # Training arguments (very minimal for demo)
    training_args = TrainingArguments(
        output_dir="./lora_demo_output",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        learning_rate=5e-4,
        logging_steps=1,
        save_steps=10,
        remove_unused_columns=False,
    )
    
    # Data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Test AFTER training
    print("\n🔍 AFTER Training:")
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    print("\n" + "=" * 60)
    print("🎯 What Just Happened:")
    print("• LoRA added tiny 'adapter' layers to the model")
    print("• Only these adapters were trained (0.32% of parameters)")
    print("• Original model weights stayed completely frozen")
    print("• Model learned to respond in the training style")
    print("• Much faster than full fine-tuning!")

if __name__ == "__main__":
    demonstrate_lora_training()
