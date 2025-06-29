import torch
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import os

# Suppress a harmless warning from the MPS backend
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- Configuration ---
# Using a model from the Qwen2 family, as seen in the paper.
# BASE_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct" # this model is not available on Hugging Face anymore
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


# --- Main Orchestrator ---
def main():
    """
    Main function for Knowledge Incorporation, adapted for Apple Silicon.
    """
    # --- 1. Set up device (MPS for Apple Silicon) ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ MPS device found. Using Apple Silicon GPU.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚠️ MPS not available. Falling back to {device}.")

    # --- 2. Set up Base Model and Tokenizer ---
    print(f"--- Setting up Base Model ({BASE_MODEL_ID}) and Tokenizer ---")

    print(f"Loading model and tokenizer on {device}...")
    # Load the base model and tokenizer, specifying dtype and device
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,  # Use half-precision for memory efficiency
    ).to(device)

    print(f"Model loaded successfully on {device}.")
    # Configure LoRA for the model
    print(f"Loading Tokenizer for {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    print("Tokenizer loaded successfully.")
    # Set the tokenizer's pad token to the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model and Tokenizer setup complete. Model is on {device}.")
    # --- 3. Loading Data ---
    passage = "The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by NASA, which succeeded in preparing and landing the first humans on the Moon from 1968 to 1972. The first man to walk on the moon was Neil Armstrong."
    question = "Who was the first person to walk on the moon?"
    ground_truth_answer = "Neil Armstrong"
    print(f"\nPASSAGE: {passage[:100]}...")

    # --- 4. Starting Outer Loop (ReST^EM Simulation) ---
    print("\n--- Starting Outer Loop (ReST^EM Simulation) ---")
    num_candidates = 3
    candidates = []
    for _ in range(num_candidates):
        # Pass the device to the generation function
        self_edit = generate_self_edit(model, tokenizer, passage, device)
        print(f"Generated Self-Edit [{_}]: {self_edit[:100]}...")
        candidates.append(self_edit)

    best_reward = -1
    best_self_edit = ""
    for i, self_edit in enumerate(candidates):
        print(f"\n--- Evaluating Candidate {i + 1} ---")

        # Pass the device to the inner loop function
        reward = inner_loop_update_and_evaluate(
            base_model_id=BASE_MODEL_ID,
            self_edit_text=self_edit,
            question=question,
            ground_truth_answer=ground_truth_answer,
            tokenizer=tokenizer,
            device=device
        )
        if reward > best_reward:
            best_reward = reward
            best_self_edit = self_edit

    # --- 5. M-Step (Policy Update Simulation) ---
    print("\n--- M-Step (Policy Update Simulation) ---")
    print("The policy model would be finetuned on the winning prompt-completion pair.")
    print(f"WINNING SELF-EDIT:\n{best_self_edit}")


# --- Helper Functions ---

def build_implication_prompt(passage):
    return f"""Let's read the following passage and produce a list of implications derived directly or indirectly from the content.
            Passage:
            {passage}
            
            Implications:
            1."""


def generate_self_edit(model, tokenizer, context, device):
    """Generates a 'self-edit' on the specified device."""
    prompt = build_implication_prompt(context)
    print(f"Generating self-edit for the passage: {context[:100]}...")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Generated prompt for self-editing: {text}...")
    # Ensure inputs are moved to the mps device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=1.0)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assistant_prompt = "assistant\n"
    self_edit_text = full_text.split(assistant_prompt)[-1].strip()
    return self_edit_text


def inner_loop_update_and_evaluate(base_model_id, self_edit_text, question, ground_truth_answer, tokenizer, device):
    """Represents the inner loop, running on MPS."""
    # Load a fresh model for the inner loop
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16
    ).to(device)

    # LoRA config for Qwen2 models
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)

    # Dataset and Trainer setup
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item

        def __len__(self):
            return len(self.encodings['input_ids'])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./seal_sft_temp",
            num_train_epochs=3,
            report_to="none",
            overwrite_output_dir=True,
            # use_mps_device=True  # Explicitly tell Trainer to use MPS
            label_names=["labels"],
        ),
        train_dataset=TextDataset([self_edit_text], tokenizer),
    )

    print("  - Finetuning on MPS...")
    trainer.train()

    print("  - Evaluating the updated model...")
    messages = [{"role": "user", "content": f"Question: {question}"}]
    eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(eval_prompt, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=20)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    model_answer = generated_text.split("assistant\n")[-1].strip()
    print(f"  - Model's generated answer: '{model_answer}'")

    reward = 1 if re.search(r'\b' + re.escape(ground_truth_answer) + r'\b', model_answer, re.IGNORECASE) else 0
    print(f"Reward: {reward}")

    # Clean up memory on MPS device
    del model, trainer
    torch.mps.empty_cache()
    return reward


if __name__ == "__main__":
    main() # Run the main function to start the Knowledge Incorporation process