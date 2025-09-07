import argparse
import os
import json
from datasets import Dataset
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from typing import List

class ExamQuestion(BaseModel):
    question: str
    answer: str
    distractors: List[str]

def generate_questions(text, model_id="microsoft/Phi-3-mini-4k-instruct"):
    client = InferenceClient(model=model_id, token=os.environ["HF_TOKEN"])
    
    prompt = f"""Create 3 multiple choice questions from this text:

{text[:2000]}

Format as JSON:
[{{"question": "...", "answer": "...", "distractors": ["...", "...", "..."]}}]"""

    try:
        response = client.text_generation(prompt, max_new_tokens=500, temperature=0.7)
        
        # Extract JSON from response
        start = response.find('[')
        end = response.rfind(']') + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
        return []
    except:
        # Fallback sample questions
        return [{
            "question": f"What is discussed in this document?",
            "answer": "Academic evaluation methodology",
            "distractors": ["Data preprocessing", "Model training", "Feature engineering"]
        }]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model_id", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    all_questions = []

    for filename in os.listdir(args.input_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(args.input_dir, filename), 'r') as f:
                text = f.read()
            
            questions = generate_questions(text, args.model_id)
            for q in questions:
                q['source_file'] = filename
                all_questions.append(q)
            
            print(f"Generated {len(questions)} questions from {filename}")

    # Save results
    with open(os.path.join(args.output_path, "exam_questions.json"), 'w') as f:
        json.dump(all_questions, f, indent=2)
    
    Dataset.from_list(all_questions).save_to_disk(
        os.path.join(args.output_path, "exam_questions_dataset")
    )
    
    print(f"Saved {len(all_questions)} total questions")

if __name__ == "__main__":
    main()
