# Domain Specific Evaluation with Argilla, Distilabel, and LightEval

Most popular benchmarks look at very general capabilities (reasoning, math, code), but have you ever needed to study more specific capabilities? 

What should you do if you need to evaluate a model on a **custom domain** relevant to your use-cases? (For example, financial, legal, medical use cases)  

This tutorial shows you the full pipeline you can follow, from creating relevant data and annotating your samples to evaluating your model on them, with the easy to use [Argilla](https://github.com/argilla-io/argilla), [distilabel](https://github.com/argilla-io/distilabel), and [lighteval](https://github.com/huggingface/lighteval). For our example, we'll focus on generating exam questions from multiple documents. 

## Project Structure

For our process, we will follow 4 steps, with a script for each: generating a dataset, annotating it, extracting relevant samples for evaluation from it, and actually evaluating models.

| Script Name | Description |
|-------------|-------------|
| generate_dataset_fixed.py | Generates exam questions from multiple text documents using a specified language model. |
| annotate_dataset.py | Creates an Argilla dataset for manual annotation of the generated exam questions. |
| create_dataset.py | Processes annotated data from Argilla and creates a Hugging Face dataset. |
| evaluation_task.py | Defines a custom LightEval task for evaluating language models on the exam questions dataset. |

## Prerequisites

### 1. Start Argilla Server
```bash
# Start Argilla using Docker
docker run -d --name argilla -p 6900:6900 argilla/argilla-quickstart:latest

# Verify it's running
docker ps | grep argilla

# Access web interface at http://localhost:6900
# Login: admin / 12345678
# API Key: admin.apikey
```

### 2. Extract Text from PDFs (if needed)
```bash
# Install PDF processing library
pip install pdfplumber

# Extract text from PDFs to create text files
python3 extract_pdf_text.py "/path/to/pdf/directory" "./text_data"
```

## Steps

### 1. Generate Dataset

The `generate_dataset_fixed.py` script uses the Hugging Face Inference API to generate exam questions based on multiple text documents. It creates questions, correct answers, and incorrect answers (known as distractors).

To run the generation:

```bash
python3 generate_dataset_fixed.py --input_dir "./text_data" --model_id "microsoft/Phi-3-mini-4k-instruct" --output_path "/path/to/output/"
```

This will create a JSON file containing the generated exam questions for all documents in the input directory.

### 2. Convert to Distiset Format

Convert the JSON output to Distiset format expected by the annotation script:

```bash
python3 create_distiset_fixed.py
```

This creates the proper Distiset format at `/path/to/exam_questions_distiset_fixed`.

### 3. Annotate Dataset

The `annotate_dataset.py` script takes the generated questions and creates an Argilla dataset for annotation. It sets up the dataset structure and populates it with the generated questions and answers, randomizing the order of answers to avoid bias.

To run the annotation process:

```bash
python3 -c "
import sys
sys.path.append('.')
exec(open('annotate_dataset.py').read().replace(
    'for exam in distiset[args.dataset_config][args.dataset_split]:',
    'for exam in distiset[args.dataset_config]:'
))
" --dataset_path "/path/to/exam_questions_distiset_fixed" --output_dataset_name "exam_questions_annotated" --argilla_api_key "admin.apikey"
```

This will create an Argilla dataset that can be used for manual review and annotation.

**Manual Annotation:**
1. Visit http://localhost:6900
2. Login with admin/12345678
3. Find your dataset "exam_questions_annotated"
4. Review and validate questions and answers

![argilla_dataset](./images/domain_eval_argilla_view.png)

### 4. Create Dataset

The `create_dataset.py` script processes the annotated data from Argilla and creates a Hugging Face dataset. It handles both suggested and manually annotated answers.

**First, create a HuggingFace repository:**
```bash
# Fix authentication issues
unset HF_TOKEN
hf auth login

# Create dataset repository
hf repo create your-dataset-name --repo-type dataset
```

**Then create the final dataset:**
```bash
env -u HF_TOKEN python3 create_dataset.py --dataset_path exam_questions_annotated --dataset_repo_id your-username/your-dataset-name --argilla_api_key admin.apikey
```

This will push the dataset to the Hugging Face Hub under the specified repository.

![hf_dataset](./images/domain_eval_dataset_viewer.png)

### 5. Evaluation Task

The `evaluation_task.py` script defines a custom LightEval task for evaluating language models on the exam questions dataset. It includes a prompt function, a custom accuracy metric, and the task configuration.

**Set environment variables to avoid threading issues:**
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

**To evaluate a model using lighteval with the custom exam questions task:**
```bash
lighteval accelerate \
    "pretrained=HuggingFaceH4/zephyr-7b-beta" \
    "community|exam_questions|0|0" \
    --custom-tasks evaluation_task.py \
    --output-dir "./evals"
```

## Troubleshooting

### Common Issues:

1. **Argilla Connection Issues:**
   - Ensure Docker container is running: `docker ps | grep argilla`
   - Use correct API key: `admin.apikey`
   - Check Argilla is accessible at http://localhost:6900

2. **HuggingFace Authentication:**
   - Remove invalid environment variable: `unset HF_TOKEN`
   - Login fresh: `hf auth login`
   - Use `env -u HF_TOKEN` prefix for commands if needed

3. **Distiset Format Issues:**
   - Ensure you convert JSON to Distiset format using `create_distiset_fixed.py`
   - Use the modified annotation script that iterates directly over the dataset

4. **LightEval Threading Issues:**
   - Set environment variables to disable parallelism
   - Try with smaller models first (e.g., `gpt2`)
   - Use `CUDA_VISIBLE_DEVICES=""` for CPU-only evaluation

### File Structure:
```
project/
├── extract_pdf_text.py          # Extract text from PDFs
├── generate_dataset_fixed.py    # Generate questions (fixed version)
├── create_distiset_fixed.py     # Convert JSON to Distiset
├── annotate_dataset.py          # Create Argilla annotation dataset
├── create_dataset.py            # Export to HuggingFace Hub
├── evaluation_task.py           # LightEval task definition
├── text_data/                   # Extracted text files
├── exam_questions.json          # Generated questions
├── exam_questions_distiset_fixed/  # Distiset format
└── evals/                       # Evaluation results
```

You can find detailed guides in lighteval wiki about each of these steps: 

- [Creating a Custom Task](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Creating a Custom Metric](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Using existing metrics](https://github.com/huggingface/lighteval/wiki/Metric-List)
