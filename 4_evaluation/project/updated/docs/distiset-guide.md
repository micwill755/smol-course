# Distiset: A Beginner's Guide

## What is a Distiset?

A **Distiset** is a special data format used by the `distilabel` library to store synthetic datasets created by AI models. Think of it as a smart container that holds your generated data along with information about how it was created.

## Why Use Distiset?

- **Organized Storage**: Keeps multiple related datasets together
- **Metadata Tracking**: Remembers which AI model generated what data
- **Reproducibility**: You can recreate the same data later
- **Tool Integration**: Works seamlessly with annotation tools like Argilla

## Basic Structure

A Distiset is like a dictionary that contains one or more datasets:

```python
Distiset({
    "dataset_1": Dataset(...),
    "dataset_2": Dataset(...),
    "dataset_3": Dataset(...)
})
```

## Real Example

Here's what our exam questions Distiset looks like:

```python
Distiset({
    "default": Dataset({
        features: ['generation', 'generation_model', 'source_file'],
        num_rows: 5
    })
})
```

## What's Inside Each Record?

Each record in the dataset contains:

```python
{
    "generation": '{"exam": [{"question": "What is AI?", "answer": "Artificial Intelligence", "distractors": ["Apple Inc", "Automated Input", "Advanced Internet"]}]}',
    "generation_model": "microsoft/Phi-3-mini-4k-instruct",
    "source_file": "ai_textbook.txt"
}
```

### Breaking it down:
- **`generation`**: The actual generated content (as JSON string)
- **`generation_model`**: Which AI model created this data
- **`source_file`**: What document was used as input

## How to Create a Distiset

### Method 1: From JSON Data
```python
import json
from datasets import Dataset
from distilabel.distiset import Distiset

# Your data
questions = [
    {
        "question": "What is Python?",
        "answer": "A programming language",
        "distractors": ["A snake", "A movie", "A car"]
    }
]

# Convert to Distiset format
records = []
for q in questions:
    record = {
        "generation": json.dumps({"exam": [q]}),
        "generation_model": "gpt-4",
        "source_file": "python_tutorial.txt"
    }
    records.append(record)

# Create Distiset
dataset = Dataset.from_list(records)
distiset = Distiset({"default": dataset})

# Save it
distiset.save_to_disk("my_distiset")
```

### Method 2: From Distilabel Pipeline
```python
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

# Pipeline automatically creates Distiset
with Pipeline() as pipeline:
    text_gen = TextGeneration(...)

distiset = pipeline.run(...)
```

## How to Use a Distiset

### Loading
```python
from distilabel.distiset import Distiset

# Load from disk
distiset = Distiset.load_from_disk("path/to/distiset")
```

### Accessing Data
```python
# Get a specific dataset
dataset = distiset["default"]
print(f"Number of records: {len(dataset)}")

# Iterate through records
for record in dataset:
    # Parse the generated content
    generation_data = json.loads(record["generation"])
    questions = generation_data["exam"]
    
    for q in questions:
        print(f"Q: {q['question']}")
        print(f"A: {q['answer']}")
```

### Converting to Other Formats
```python
# To regular HuggingFace Dataset
hf_dataset = distiset["default"]

# To pandas DataFrame
df = distiset["default"].to_pandas()

# To JSON
data = [json.loads(record["generation"]) for record in distiset["default"]]
```

## File Structure on Disk

When saved, a Distiset creates this structure:
```
my_distiset/
├── default/                    # Dataset name
│   ├── dataset_info.json      # Dataset metadata
│   ├── data-00000-of-00001.parquet  # Actual data
│   └── state.json             # Dataset state
├── distiset.yaml              # Distiset configuration
└── artifacts/                 # Additional files (if any)
```

## Common Use Cases

### 1. Question Generation
```python
# Generate exam questions from documents
distiset = generate_questions_pipeline.run(documents)
```

### 2. Data Annotation
```python
# Send to Argilla for human review
annotate_dataset.py --dataset_path my_distiset --output_dataset_name annotated
```

### 3. Model Evaluation
```python
# Use for benchmarking models
evaluation_task.py --dataset my_distiset
```

## Best Practices

1. **Use descriptive names** for your datasets within the Distiset
2. **Include metadata** like model names and source files
3. **Save regularly** during long generation processes
4. **Version your Distisets** for reproducibility
5. **Document your generation process** in the artifacts folder

## Troubleshooting

### Common Issues:
- **"Column 'train' doesn't exist"**: Your dataset doesn't have splits, access directly with `distiset["dataset_name"]`
- **JSON parsing errors**: Check that your `generation` field contains valid JSON
- **Loading failures**: Ensure all required files are present in the directory

### Quick Fixes:
```python
# Check what's in your Distiset
print("Available datasets:", list(distiset.keys()))
print("Dataset info:", distiset["default"])

# Inspect a record
print("Sample record:", distiset["default"][0])
```

## Summary

Distiset is distilabel's way of packaging AI-generated data with context. It's like a smart folder that remembers:
- What data was generated
- How it was generated  
- When it was generated
- What it can be used for

This makes your AI data pipelines more organized, reproducible, and ready for downstream tasks like annotation and evaluation.
