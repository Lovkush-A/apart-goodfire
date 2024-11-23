from goodfire.variants import Variant
from goodfire.api.client import Client
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import numpy as np
from typing import List, Tuple, TypeAlias
from tqdm import tqdm

def get_completion(client: Client, variant: Variant, prompt: str, max_tokens: int = 50) -> str:
    completion = ""
    for token in client.chat.completions.create(
        [
            {"role": "user", "content": prompt}
        ],
        model=variant,
        stream=True,
        max_completion_tokens=max_tokens,
    ):
        completion += token.choices[0].delta.content
    
    return completion

# create alias for messages
Messages: TypeAlias = list[dict[str, str]]

def get_activations(
    model: HookedTransformer,
    messages: Messages,
    layer: int = -1,
    n_activations: int = 5,
) -> torch.Tensor:
    """Get activations at a specific layer and position."""
    tokens = model.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # find first time assistant starts generating completions. assistant token is 78191
    assistant_token_pos = torch.nonzero(tokens == 78191, as_tuple=True)[1][0].item()

    # assert that the next three tokens are 128007, 271, 0
    assert tokens[0, assistant_token_pos + 1] == 128007
    assert tokens[0, assistant_token_pos + 2] == 271

    start_pos = assistant_token_pos + 3
    end_pos = start_pos + n_activations

    _, cache = model.run_with_cache(tokens[0:end_pos])

    # Get residual stream activations at specified layer
    activations = cache["resid_post", layer][0]

    # Average activations across positions, from start_pos to end_pos
    activations = activations[start_pos:end_pos]

    return activations.mean(dim=0)

def compute_steering_vector(
    model: HookedTransformer,
    pairs: List[Tuple[Messages, Messages]],
    layer: int = -1,
    n_activations: int = 5
) -> torch.Tensor:
    """
    Compute steering vector from contrastive pairs.
    
    Args:
        model: HookedTransformer model
        pairs: List of (x, y) sentence pairs
        layer: Layer to extract activations from (-1 for final layer)
        n_activations: number of tokens whose activations to average.
    
    Returns:
        Steering vector as torch tensor
    """
    # Get activations for each sentence in pairs
    x_activations = []
    y_activations = []
    
    for x, y in tqdm(pairs):
        x_act = get_activations(model, x, layer, n_activations)
        y_act = get_activations(model, y, layer, n_activations)
        
        x_activations.append(x_act)
        y_activations.append(y_act)
    
    # Stack activations
    x_activations = torch.stack(x_activations)
    y_activations = torch.stack(y_activations)
    
    # Compute difference vectors
    diff_vectors = y_activations - x_activations
    
    # Average difference vectors to get steering vector
    steering_vector = diff_vectors.mean(dim=0)
    
    # Normalize steering vector
    steering_vector = steering_vector / torch.norm(steering_vector)
    
    return steering_vector

def apply_steering(
    model: HookedTransformer,
    messages: Messages,
    steering_vector: torch.Tensor,
    n_tokens: int,
    layer: int = -1,
    strength: float = 1.0
) -> str:
    """
    Apply steering vector to generate text from a prefix.
    
    Args:
        model: HookedTransformer model
        prefix: Input text prefix to start generation
        steering_vector: Computed steering vector
        n_tokens: Number of tokens to generate
        layer: Layer to apply steering at
        strength: Scaling factor for steering vector
    
    Returns:
        Generated text after applying steering
    """
    tokens = model.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    def hook_fn(activations, hook):
        # Add scaled steering vector to residual stream
        return activations + strength * steering_vector
    
    # Register hook at specified layer
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    generated_tokens = tokens
    for _ in range(n_tokens):
        logits = model.run_with_hooks(
            generated_tokens,
            fwd_hooks=[(hook_name, hook_fn)]
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
    
    # Convert generated tokens to text
    return model.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)