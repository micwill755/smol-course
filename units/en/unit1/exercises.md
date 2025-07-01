# Exercises

This section includes exercises that will help you apply the concepts of chat templates and supervised fine-tuning.

## Exploring Chat Templates with SmolLM2

Chat templates help structure interactions between users and AI models, ensuring consistent and contextually appropriate responses.

### Setup

First, let's set up the environment.

```python
# Install the requirements in Google Colab
# !pip install transformers datasets trl huggingface_hub

# Authenticate to Hugging Face
from huggingface_hub import login

login()

# for convenience you can create an environment variable containing your hub token as HF_TOKEN
```

```python
# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
import torch
```

### SmolLM2 Chat Template

Let's explore how to use a chat template with the `SmolLM2` model. We'll define a simple conversation and apply the chat template.

```python
# Dynamically set the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
```

```python
# Define messages for SmolLM2
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {
        "role": "assistant",
        "content": "I'm doing well, thank you! How can I assist you today?",
    },
]
```

The tokenizer represents the conversation as a string with special tokens to describe the role of the user and the assistant.

```python
input_text = tokenizer.apply_chat_template(messages, tokenize=False)

print("Conversation with template:", input_text)
```

**Output:**
```
Conversation with template: <|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing well, thank you! How can I assist you today?<|im_end|>
```

Note that the conversation is represented as above but with a further assistant message.

```python
input_text = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True
)

print("Conversation decoded:", tokenizer.decode(token_ids=input_text))
```

**Output:**
```
Conversation decoded: <|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing well, thank you! How can I assist you today?<|im_end|>
<|im_start|>assistant
```

Of course, the tokenizer also tokenizes the conversation and special token as ids that relate to the model's vocabulary.

```python
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

print("Conversation tokenized:", input_text)
```

**Output:**
```
Conversation tokenized: [1, 4093, 198, 19556, 28, 638, 359, 346, 47, 2, 198, 1, 520, 9531, 198, 57, 5248, 2567, 876, 28, 9984, 346, 17, 1073, 416, 339, 4237, 346, 1834, 47, 2, 198, 1, 520, 9531, 198]
```

### Exercise: Process a dataset for SFT

Take a dataset from the Hugging Face hub and process it for SFT.

**Difficulty Levels**

*   🐢 Convert the `HuggingFaceTB/smoltalk` dataset into chatml format.
*   🐕 Convert the `openai/gsm8k` dataset into chatml format.

```python
from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")


def process_dataset(sample):
    # TODO: 🐢 Convert the sample into a chat format
    # use the tokenizer's method to apply the chat template
    return sample


ds = ds.map(process_dataset)
```

```python
ds = load_dataset("openai/gsm8k", "main")


def process_dataset(sample):
    # TODO: 🐕 Convert the sample into a chat format

    # 1. create a message format with the role and content

    # 2. apply the chat template to the samples using the tokenizer's method

    return sample


ds = ds.map(process_dataset)
```

## Supervised Fine-Tuning with SFTTrainer

This section demonstrates how to fine-tune the `HuggingFaceTB/SmolLM2-135M` model using the `SFTTrainer` from the `trl` library.

### Exercise: Fine-Tuning SmolLM2 with SFTTrainer

Take a dataset from the Hugging Face hub and finetune a model on it.

**Difficulty Levels**

*   🐢 Use the `HuggingFaceTB/smoltalk` dataset
*   🐕 Try out the `bigcode/the-stack-smol` dataset and finetune a code generation model on a specific subset `data/python`.
*   🦁 Select a dataset that relates to a real world use case your interested in

```python
# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Set up the chat format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "SmolLM2-FT-MyDataset"
finetune_tags = ["smol-course", "module_1"]
```

### Generate with the base model

Here we will try out the base model which does not have a chat template.

```python
# Let's test the base model before training
prompt = "Write a haiku about programming"

# Format with template
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate response
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print("Before training:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Dataset Preparation

We will load a sample dataset and format it for training. The dataset should be structured with input-output pairs, where each input is a prompt and the output is the expected response from the model.

TRL will format input messages based on the model's chat templates. They need to be represented as a list of dictionaries with the keys: `role` and `content`.

```python
# Load a sample dataset
from datasets import load_dataset

# TODO: define your dataset and config using the path and name parameters
ds = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")
```

🦁 If your dataset is not in a format that TRL can convert to the chat template, you will need to process it. Refer to the [module](../chat_templates.md)

### Configuring the SFTTrainer

The `SFTTrainer` is configured with various parameters that control the training process. These include the number of training steps, batch size, learning rate, and evaluation strategy. Adjust these parameters based on your specific requirements and computational resources.

```python
# Configure the SFTTrainer
sft_config = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,  # Adjust based on dataset size and desired training duration
    per_device_train_batch_size=4,  # Set according to your GPU memory capacity
    learning_rate=5e-5,  # Common starting point for fine-tuning
    logging_steps=10,  # Frequency of logging training metrics
    save_steps=100,  # Frequency of saving model checkpoints
    evaluation_strategy="steps",  # Evaluate the model at regular intervals
    eval_steps=50,  # Frequency of evaluation
    use_mps_device=(
        True if device == "mps" else False
    ),  # Use MPS for mixed precision training
    hub_model_id=finetune_name,  # Set a unique name for your model
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],
    tokenizer=tokenizer,
    eval_dataset=ds["test"],
)

# TODO: 🦁 🐕 align the SFTTrainer params with your chosen dataset. For example, if you are using the `bigcode/the-stack-smol` dataset, you will need to choose the `content` column
```

### Training the Model

With the trainer configured, we can now proceed to train the model.

```python
# Train the model
trainer.train()

# Save the model
trainer.save_model(f"./{finetune_name}")
```

```python
trainer.push_to_hub(tags=finetune_tags)
```

### Bonus Exercise: Generate with fine-tuned model

🐕 Use the fine-tuned to model generate a response, just like with the base example.

```python
# Test the fine-tuned model on the same prompt

# Let's test the base model before training
prompt = "Write a haiku about programming"

# Format with template
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate response
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

# TODO: use the fine-tuned to model generate a response, just like with the base example.
``` 