import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm

# Import our custom models
from models import BaselineGPT2, CoconutGPT2

import logging
logging.basicConfig(level=logging.INFO)


# A custom data collator to handle the multi-stage curriculum
class CurriculumDataCollator:
    def __init__(self, tokenizer, mode='cot', stage=0, c_thoughts_per_step=2):
        self.tokenizer = tokenizer
        self.mode = mode
        self.stage = stage
        self.c = c_thoughts_per_step  # Hyperparameter c from the paper

    def __call__(self, batch):
        # Dynamically create the input text based on the training mode and stage
        processed_batch = []
        for item in batch:
            question = item['question']
            steps = item['steps']
            answer = item['answer']

            # The number of steps to replace with thoughts depends on the stage
            num_steps_to_replace = self.stage

            # --- Construct the input based on training mode ---
            if self.mode == 'cot':
                # Full chain-of-thought
                text = f"{question} {' '.join(steps)} #### {answer}"
            elif self.mode == 'no-cot':
                # Question and answer only
                text = f"{question} #### {answer}"
            elif self.mode == 'icot':
                # Internalized CoT: remove first k steps
                remaining_steps = ' '.join(steps[num_steps_to_replace:])
                text = f"{question} {remaining_steps} #### {answer}"
            elif self.mode == 'coconut':
                num_latent_thoughts = num_steps_to_replace * self.c
                remaining_steps = ' '.join(steps[num_steps_to_replace:])
                # We use pad tokens as placeholders for thoughts, the model handles the rest
                thought_placeholders = ' '.join([self.tokenizer.pad_token] * num_latent_thoughts)
                text = f"{question} <bot> {thought_placeholders} <eot> {remaining_steps} #### {answer}"
            # Add other modes like 'pause' etc. here if needed

            processed_batch.append(text + self.tokenizer.eos_token)

        # Tokenize the batch
        tokenized_batch = self.tokenizer(
            processed_batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Create labels for language modeling loss
        # The labels are the same as the input_ids
        tokenized_batch['labels'] = tokenized_batch['input_ids'].clone()

        # --- Masking the loss ---
        # The paper states to mask loss on questions and latent thoughts
        # For simplicity in this sanity check, we only mask the padding tokens in labels.
        # A more rigorous implementation would mask the question tokens as well.
        labels = tokenized_batch['labels']
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        tokenized_batch['labels'] = labels

        return tokenized_batch


def train(args):
    """Main training function."""

    # --- 1. Initialize Model and Tokenizer ---
    if args.mode == 'coconut':
        model = CoconutGPT2(model_name=args.base_model)
    else:  # cot, no-cot, icot, etc.
        model = BaselineGPT2(model_name=args.base_model)

    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model.to(device)

    # --- 2. Load Data ---
    with open(f"data/train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]

    # --- 3. Training Loop ---
    logging.info(f"--- Starting Training for mode: {args.mode} ---")

    # Determine the number of stages based on the mode
    # For GSM8k, the paper uses 3 stages after the initial one
    num_stages = 4 if args.mode in ['coconut', 'icot'] else 1

    for stage in range(num_stages):
        # Per the paper, reset optimizer state when stages switch
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        # Setup the data collator for the current stage
        collator = CurriculumDataCollator(tokenizer, mode=args.mode, stage=stage)
        dataloader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collator)

        num_training_steps = args.epochs_per_stage * len(dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps), desc=f"Stage {stage + 1}/{num_stages}")

        model.train()
        for epoch in range(args.epochs_per_stage):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Log memory usage on MPS
                if device.type == "mps":
                    allocated_memory = torch.mps.current_allocated_memory()
                    logging.info(f"Step {progress_bar.n}: MPS memory allocated: {allocated_memory / (1024 ** 2):.2f} MB")

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

    # --- 4. Save Model ---
    output_dir = f"saved_models/{args.mode}"
    os.makedirs(output_dir, exist_ok=True)
    model.gpt2.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model saved to {output_dir}")


# Train the CoT baseline
# python train.py --mode cot --batch_size 4 --epochs_per_stage 6

# Train the No-CoT baseline
# python train.py --mode no-cot --batch_size 4 --epochs_per_stage 6

# Train the COCONUT model with its curriculum:
# python train.py --mode coconut --batch_size 4 --epochs_per_stage 3

# Train the iCoT model
# python train.py --mode icot --batch_size 4 --epochs_per_stage 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reasoning models for GSM8k.")
    parser.add_argument("--mode", type=str, choices=['cot', 'no-cot', 'icot', 'coconut'],
                        help="The training strategy to use.", default="cot")
    parser.add_argument("--base_model", type=str, default="gpt2", help="The base transformer model to use.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--epochs_per_stage", type=int, default=3,
                        help="Number of epochs to train for each curriculum stage.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")

    args = parser.parse_args()
    train(args)