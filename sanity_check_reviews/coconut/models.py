import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class BaselineGPT2(nn.Module):
    """
    A baseline model using GPT-2. This model can be used for:
    - CoT (Chain-of-Thought): by training on questions + reasoning steps + answers.
    - No-CoT: by training on questions + answers.
    - iCoT: by using a specific training curriculum that gradually removes reasoning steps.

    The model architecture itself is a standard GPT-2 LM Head Model.
    The differences are in the data fed during training.
    """

    def __init__(self, model_name='gpt2', special_tokens=None):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Add any special tokens if provided
        if special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            self.gpt2.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.gpt2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, input_ids, max_length):
        return self.gpt2.generate(input_ids, max_length=max_length)


class CoconutGPT2(nn.Module):
    """
    Implementation of the COCONUT model.
    This model reasons in a continuous latent space by feeding the last hidden state
    of a 'thought' step back as the input embedding for the next step.
    """

    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Add special tokens for latent reasoning mode
        special_tokens_dict = {'additional_special_tokens': ['<bot>', '<eot>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.gpt2.resize_token_embeddings(len(self.tokenizer))

        self.bot_token_id = self.tokenizer.convert_tokens_to_ids('<bot>')
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids('<eot>')

        # The pad token is required for batching
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass with latent mode logic.
        """
        # Find the start and end of latent mode for each item in the batch
        # This implementation assumes a single <bot> and <eot> per sequence for simplicity
        try:
            bot_positions = (input_ids == self.bot_token_id).nonzero(as_tuple=True)[1]
            eot_positions = (input_ids == self.eot_token_id).nonzero(as_tuple=True)[1]
            start_latent_idx = bot_positions[0] + 1
            end_latent_idx = eot_positions[0]
        except IndexError:
            # If no <bot> token, run a standard forward pass (language mode)
            return self.gpt2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Get the initial token embeddings
        inputs_embeds = self.gpt2.transformer.wte(input_ids)

        # Enter Latent Mode
        # Sequentially generate continuous thoughts
        current_embeds = inputs_embeds
        for t in range(start_latent_idx, end_latent_idx):
            # Pass all embeddings up to the current thought token
            outputs = self.gpt2.transformer(inputs_embeds=current_embeds[:, :t, :])

            # The last hidden state of the previous token becomes the "continuous thought"
            last_hidden_state = outputs.last_hidden_state

            # This "thought" is used directly as the input embedding for the current step [cite: 9, 10]
            # We update the embedding in our input tensor
            current_embeds[:, t, :] = last_hidden_state[:, -1, :]

        # After the latent loop, compute the final output with the modified embeddings
        outputs = self.gpt2(inputs_embeds=current_embeds, attention_mask=attention_mask, labels=labels)

        return outputs

    def generate(self, input_ids, max_length):
        """
        Custom generation function for COCONUT to handle latent mode during inference.
        """
        self.gpt2.eval()
        device = input_ids.device

        # --- Process the initial prompt with latent thoughts ---
        # Get the initial embeddings
        inputs_embeds = self.gpt2.transformer.wte(input_ids)

        # Find latent mode boundaries
        try:
            bot_positions = (input_ids == self.bot_token_id).nonzero(as_tuple=True)[1]
            eot_positions = (input_ids == self.eot_token_id).nonzero(as_tuple=True)[1]
            start_latent_idx = bot_positions[0] + 1
            end_latent_idx = eot_positions[0]
        except IndexError:
            # If no latent mode in prompt, use standard generation
            return self.gpt2.generate(input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)

        # Run the latent thought process
        current_embeds = inputs_embeds
        for t in range(start_latent_idx, end_latent_idx):
            outputs = self.gpt2.transformer(inputs_embeds=current_embeds[:, :t, :])
            last_hidden_state = outputs.last_hidden_state
            current_embeds[:, t, :] = last_hidden_state[:, -1, :]

        # Get the hidden states after the prompt has been processed
        outputs = self.gpt2.transformer(inputs_embeds=current_embeds)
        past_key_values = outputs.past_key_values

        # --- Standard autoregressive decoding after latent thoughts ---
        # Start decoding from the last token of the input
        generated_ids = input_ids

        for _ in range(max_length - input_ids.shape[1]):
            # Get the logits for the very last token
            last_token_id = generated_ids[:, -1].unsqueeze(-1)
            last_token_embeds = self.gpt2.transformer.wte(last_token_id)

            outputs = self.gpt2(inputs_embeds=last_token_embeds, past_key_values=past_key_values)

            # Get the predicted next token (greedy decoding)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append the new token to our sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            # Update past_key_values for the next step
            past_key_values = outputs.past_key_values

            # Stop if we generate the end-of-sequence token
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        return generated_ids