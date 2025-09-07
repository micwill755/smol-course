#!/usr/bin/env python3
import json
from datasets import Dataset
from distilabel.distiset import Distiset

def create_distiset_from_json(json_path, output_path):
    """Convert JSON questions to Distiset format expected by annotate_dataset.py"""
    
    # Load JSON data
    with open(json_path, 'r') as f:
        questions = json.load(f)
    
    # Convert to the format expected by annotate_dataset.py
    records = []
    for q in questions:
        # Create the generation field as JSON string
        generation_data = {
            "exam": [{
                "question": q["question"],
                "answer": q["answer"],
                "distractors": q["distractors"]
            }]
        }
        
        record = {
            "generation": json.dumps(generation_data),
            "generation_model": "microsoft/Phi-3-mini-4k-instruct",
            "source_file": q.get("source_file", "unknown")
        }
        records.append(record)
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(records)
    
    # Create Distiset (collection of datasets)
    distiset = Distiset({
        "default": dataset
    })
    
    # Save to disk
    distiset.save_to_disk(output_path)
    print(f"Created Distiset at {output_path}")
    return distiset

if __name__ == "__main__":
    create_distiset_from_json(
        "/Users/michaelwilliams/Documents/code/deep learning/hugging-face/smol/4. Evaluation/exam_questions.json",
        "/Users/michaelwilliams/Documents/code/deep learning/hugging-face/smol/4. Evaluation/exam_questions_distiset"
    )
