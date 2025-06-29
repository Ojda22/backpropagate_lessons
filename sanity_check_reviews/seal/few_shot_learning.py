import torch
import json
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
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


# --- M-Step Policy Update Function ---
def m_step_policy_update(model, tokenizer, prompt, winning_completion_json, device):
    """
    Performs the M-Step by finetuning the policy model on the winning
    prompt and JSON configuration.
    """
    print("\n--- Performing M-Step: Updating policy model ---")

    # The training data is the prompt concatenated with the best JSON completion.
    # This teaches the model: "Given this ARC task, generate this kind of JSON."
    winning_completion_str = json.dumps(winning_completion_json, indent=2)
    training_text = prompt + "\n" + winning_completion_str

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
            output_dir="./seal_arc_policy_update_temp",
            num_train_epochs=1,
            learning_rate=2e-5,
            report_to="none",
            overwrite_output_dir=True,
            use_mps_device=True
        ),
        train_dataset=TextDataset([training_text], tokenizer),
    )

    print("  - Finetuning the base model on the winning JSON configuration...")
    trainer.train()

    print("--- Policy model has been updated. ---")
    return model


# --- Main Orchestrator ---
def main():
    """
    Main function for ARC Few-Shot Learning, adapted for Apple Silicon.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ MPS device found. Using Apple Silicon GPU.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚠️ MPS not available. Falling back to {device}.")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
        ).to(device) # we use LLama 3.2 1B Instruct model to replicate the paper
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you are logged into Hugging Face and have accepted the Llama 3.2 license.")
        return

    lora_config = LoraConfig(
        r=16, lora_alpha=16, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )  # we setup lora fentuning configuration for the model
    policy_model = get_peft_model(base_model, lora_config) # we apply lora configuration to the base model

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token  # we get the tokenizer for the model and set the pad token to eos token

    arc_task_context = """
[Task Description]
You will be given a few input/output pairs demonstrating a transformation rule.
Apply the same rule to the test input to generate the test output.

[Training Example 1]
Input: "111"
Output: "121"

[Training Example 2]
Input: "444"
Output: "424"

[Test Input]
Input: "888"
"""
    ground_truth_test_output = "828"
    print("\n--- ARC Task Loaded ---")
    # This is the sanity check for the ARC task, we define the task context and ground truth output

    print("\n--- Starting Outer Loop (ReST^EM Simulation) ---")
    num_candidates = 3
    candidates = []
    for i in range(num_candidates):
        print(f"  - Generating candidate configuration {i + 1}...")
        # we aim to generate 3 candidate configurations for the ARC task
        config_json = generate_arc_self_edit(policy_model, tokenizer, arc_task_context, device)
        if config_json: # we check if the generated configuration is valid JSON
            candidates.append(config_json)

    best_reward = -1
    best_config = None
    for i, config in enumerate(candidates):
        print(f"\n--- Evaluating Candidate Config {i + 1} ---")
        print(f"CONFIG: {config}")

        # for each candidate configuration, we run the inner loop to evaluate its performance
        reward = arc_inner_loop(
            base_model_id=BASE_MODEL_ID,
            tokenizer=tokenizer,
            arc_config=config,
            arc_task_context=arc_task_context,
            ground_truth_test_output=ground_truth_test_output,
            device=device
        )
        if reward > best_reward:
            best_reward = reward
            best_config = config

    # --- M-Step (Policy Update Simulation) ---
    if best_config:
        arc_prompt = build_arc_prompt(arc_task_context)
        updated_policy_model = m_step_policy_update(
            model=policy_model,
            tokenizer=tokenizer,
            prompt=arc_prompt,
            winning_completion_json=best_config,
            device=device
        )

        # --- Verification Step ---
        print("\n--- Verifying Policy Update ---")
        print("Generating a new self-edit with the *updated* policy model...")
        new_config = generate_arc_self_edit(updated_policy_model, tokenizer, arc_task_context, device)

        print("\nOriginal Winning Configuration:")
        print(json.dumps(best_config, indent=2))
        print("\nNew Configuration from Updated Model:")
        print(json.dumps(new_config, indent=2) if new_config else "Failed to generate valid JSON.")
    else:
        print("\n--- No winning configuration found to update policy. ---")


def build_arc_prompt(arc_context):
    return f"""{arc_context}

[Instructions]
You are configuring a model training pipeline. Respond with a valid JSON object specifying the data augmentations and training configuration. Do not include any explanation.

- Data Generation:
  - "use_basic_augmentations": (boolean) Use basic rotational/flip augmentations.
- Training Configuration:
  - "strategy": (string) "train_using_all_tokens" or "train_using_output_tokens".
  - "learning_rate": (float) e.g., 5e-5, 1e-4.
  - "num_train_epochs": (integer) e.g., 1, 2, 3.

[Your Configuration]
"""


def generate_arc_self_edit(model, tokenizer, arc_context, device):
    prompt = build_arc_prompt(arc_context)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.6)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        # Extract the JSON part from the generated text
        json_str = re.search(r'\{.*\}', full_text, re.DOTALL).group(0)
        return json.loads(json_str)
    except (AttributeError, json.JSONDecodeError):
        print("  - Warning: Failed to generate or parse valid JSON.")
        return None


def arc_inner_loop(base_model_id, tokenizer, arc_config, arc_task_context, ground_truth_test_output, device):
    """Represents the inner loop for the ARC task, running on MPS."""
    temp_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16).to(device)

    lora_config = LoraConfig(r=16, lora_alpha=16, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    temp_model_adapted = get_peft_model(temp_model, lora_config)

    training_examples = {"Input: \"111\"\nOutput:": "\"121\"", "Input: \"444\"\nOutput:": "\"424\""}

    # Simulate data augmentation based on the configuration
    if arc_config.get("data_generation", {}).get("use_basic_augmentations", False):
        print("  - Applying simulated data augmentation...")
        augmented_examples = {k.replace('"', '')[::-1] + '"': v.replace('"', '')[::-1] + '"' for k, v in
                              training_examples.items()}
        # Reverse the keys and values to simulate augmentation
        training_examples.update(augmented_examples) # we augment the training examples with reversed strings
    train_dataset_text = [f"{k} {v}" for k, v in training_examples.items()] # we prepare the training dataset text

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item

        def __len__(self):
            return len(self.encodings['input_ids'])

    training_params = arc_config.get("training", {}) # we extract the training parameters from the configuration
    lr = float(training_params.get("learning_rate", 1e-4)) # we get the learning rate from the configuration, default to 1e-4
    epochs = int(training_params.get("num_train_epochs", 2)) # we get the number of epochs from the configuration, default to 2

    trainer = Trainer(
        model=temp_model_adapted,
        args=TrainingArguments(
            output_dir="./seal_arc_temp", num_train_epochs=epochs, learning_rate=lr,
            per_device_train_batch_size=1, report_to="none", overwrite_output_dir=True,
            # use_mps_device=True
            label_names=["labels"],
        ),
        train_dataset=TextDataset(train_dataset_text, tokenizer),
    )

    print(f"  - Finetuning a temporary model on MPS with lr={lr}, epochs={epochs}...")
    trainer.train()

    print("  - Evaluating the temporary model...")
    eval_prompt = f"{arc_task_context}\n[Test Output]\nOutput:"
    messages = [{"role": "user", "content": eval_prompt}]
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_prompt, return_tensors="pt").to(device)

    outputs = temp_model_adapted.generate(**inputs, max_new_tokens=10)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_answer = generated_text.split("assistant\n")[-1].strip().replace('"', '')
    print(f"  - Model's generated answer: '{model_answer}'")

    reward = 1 if model_answer == ground_truth_test_output else 0
    print(f"Reward: {reward}")

    del temp_model, temp_model_adapted, trainer
    torch.mps.empty_cache()
    return reward


if __name__ == "__main__":
    main() # Run the main function to start the ARC Few-Shot Learning process