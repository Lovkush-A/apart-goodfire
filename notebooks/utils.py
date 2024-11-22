from goodfire.variants import Variant
from goodfire.api.client import Client

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