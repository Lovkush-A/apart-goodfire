from tqdm import tqdm
from typing import List, Tuple, TypeAlias

import torch

from goodfire.variants import Variant
from goodfire.api.client import Client
from transformer_lens import HookedTransformer

def get_completion(client: Client, variant: Variant, prompt: str, max_tokens: int = 50) -> str:
    """Get text completion from the Goodfire API.
    
    Code copied from Goodfire notebooks examples.

    Parameters
    ----------
    client: Client
        Goodfire API client
    variant: Variant
        Goodfire variants are models with hooks/modifications included for steering.
    prompt: str
        Text prompt to generate completion from.
        This will be the user input to the assistant.
    max_tokens: int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        Generated text completion.
    """
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
    layer: int = 1,
    n_activations: int = 5,
) -> torch.Tensor:
    """Get average activations at specified layer for the first n_activations assistants' tokens.
    
    WARNING:
    - just read the code to understand what this function does. Should be easy to read.
    - This is brittle and has various assumptions baked in. E.g. likely only works on specific
      LLama 8b instruct model.

    Parameters
    ----------
    model: HookedTransformer
        Model to extract activations from.
        Assume that the model is an Instruct model, whose tokenzier has `apply_chat_template` method.
        Assumes specific tokenization IDs, which work for Llama 8b Instruct.
    messages: Messages
        List of chat messages to generate activations from.
        Note that the code implicitly assume there is only one assistant message in the chat.
    layer: int
        Layer of model to extract activations from.
    n_activations: int
        Number of tokens whose activations to average.
        E.g. value of 1 means you take the activations of the first assistant token.
    
    Returns
    -------
    torch.Tensor
        Average activations at specified layer for the first n_activations assistants' tokens.
    """
    # tokenize
    tokens = model.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # find first time assistant starts generating completions. `assistant` token is 78191
    assistant_token_pos = torch.nonzero(tokens == 78191, as_tuple=True)[1][0].item()

    # assert that the next three tokens are 128007, 271
    # 128007 is special token to indicate 'end of message type' and 271 are newlines.
    assert tokens[0, assistant_token_pos + 1] == 128007
    assert tokens[0, assistant_token_pos + 2] == 271

    # Assistant's response starts after the newline token
    start_pos = assistant_token_pos + 3
    end_pos = start_pos + n_activations

    _, cache = model.run_with_cache(tokens[:, 0:end_pos])

    # Get residual stream activations at specified layer
    activations = cache["resid_post", layer][0]

    # Average activations across positions, from start_pos to end_pos
    activations = activations[start_pos:end_pos]

    return activations.mean(dim=0)

def compute_steering_vector(
    model: HookedTransformer,
    pairs: List[Tuple[Messages, Messages]],
    layer: int = 1,
    n_activations: int = 5
) -> torch.Tensor:
    """
    Compute steering vector from contrastive pairs.

    Most of the subtleties/logic are in the `get_activations` function.

    Parameters
    ----------
    model: HookedTransformer
        Model to extract activations from.
    pairs: List[Tuple[Messages, Messages]]
        List of contrastive pairs of messages to compute steering vector from.
    layer: int
        Layer of model to extract activations from
    n_activations: int
        number of tokens whose activations to average.
    
    Returns
    -------
    torch.Tensor
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
    
    Parameters
    ----------
    model: HookedTransformer model
    prefix: Input text prefix to start generation
    steering_vector: Computed steering vector
    n_tokens: Number of tokens to generate
    layer: Layer to apply steering at
    strength: Scaling factor for steering vector
    
    Returns
    -------
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