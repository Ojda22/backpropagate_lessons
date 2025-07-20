import os
import re
from datasets import load_dataset

import logging
logging.basicConfig(level=logging.INFO)

def parse_solution(solution):
    """
    Parses the solution string to extract reasoning steps and the final answer.
    The reasoning steps are enclosed in <<...>>.
    """
    steps = re.findall(r'<<.*?>>', solution)
    answer = solution.split('####')[-1].strip()
    return steps, answer


def preprocess_gsm8k():
    """
    Downloads the GSM8k dataset and preprocesses it.
    """
    logging.info("Downloading and preprocessing GSM8k dataset...")

    # Load the dataset from Hugging Face
    dataset = load_dataset("gsm8k", "main")

    # Create a directory to save the preprocessed data
    if not os.path.exists("data"):
        os.makedirs("data")

    for split in dataset.keys():
        processed_samples = []
        for example in dataset[split]:
            steps, answer = parse_solution(example['answer'])
            processed_samples.append({
                'question': example['question'],
                'steps': steps,
                'answer': answer
            })

        # Save the processed data to a file
        import json
        with open(f"data/{split}.jsonl", "w") as f:
            for sample in processed_samples:
                f.write(json.dumps(sample) + "\n")

    logging.info("Preprocessing complete. Data saved in the 'data' directory.")


if __name__ == "__main__":
    preprocess_gsm8k()