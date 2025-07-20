import argparse
import json
import torch
from tqdm import tqdm
import re

import logging
logging.basicConfig(level=logging.INFO)

from models import BaselineGPT2, CoconutGPT2


def extract_answer(text):
    """Extracts the final numerical answer from the generated text."""
    # Look for the answer after '####'
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return match.group(1).strip()
    # Fallback: return the last number in the string
    numbers = re.findall(r'\d+', text)
    return numbers[-1] if numbers else ""


def evaluate(args):
    """Main evaluation function."""
    logging.info(f"--- Evaluating model from: {args.model_path} ---")

    # --- 1. Load Model and Tokenizer ---
    if args.mode == 'coconut':
        model = CoconutGPT2()
        model.gpt2 = model.gpt2.from_pretrained(args.model_path)
        model.tokenizer = model.tokenizer.from_pretrained(args.model_path)
    else:
        model = BaselineGPT2()
        model.gpt2 = model.gpt2.from_pretrained(args.model_path)
        model.tokenizer = model.tokenizer.from_pretrained(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --- 2. Load Test Data ---
    with open("data/test.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]

    # --- 3. Generate and Compare ---
    correct_predictions = 0
    total_predictions = 0

    for item in tqdm(test_data, desc="Evaluating"):
        question = item['question']
        ground_truth_answer = item['answer']

        prompt = question
        # For COCONUT, we construct the prompt to trigger latent mode
        if args.mode == 'coconut':
            # Use the number of thoughts from the final training stage
            # In our setup, final stage (stage 3) has 3 * c thoughts
            num_latent_thoughts = 3 * 2  # c=2 for GSM8k
            thought_placeholders = ' '.join([model.tokenizer.pad_token] * num_latent_thoughts)
            prompt = f"{question} <bot> {thought_placeholders} <eot>"

        inputs = model.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(inputs.input_ids, max_length=256)

        generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        predicted_answer = extract_answer(generated_text)

        if predicted_answer == ground_truth_answer:
            correct_predictions += 1
        total_predictions += 1

    # --- 4. Report Accuracy ---
    accuracy = (correct_predictions / total_predictions) * 100
    logging.info(f"\n--- Evaluation Complete ---")
    logging.info(f"Correct Predictions: {correct_predictions}")
    logging.info(f"Total Predictions:   {total_predictions}")
    logging.info(f"Accuracy: {accuracy:.2f}%")

# Evaluate the CoT model
# python evaluate.py --model_path saved_models/cot --mode cot

# Evaluate the COCONUT model
# python evaluate.py --model_path saved_models/coconut --mode coconut

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reasoning models.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory.")
    parser.add_argument("--mode", type=str, required=True, choices=['cot', 'no-cot', 'icot', 'coconut'],
                        help="The mode the model was trained in.")

    args = parser.parse_args()
    evaluate(args)