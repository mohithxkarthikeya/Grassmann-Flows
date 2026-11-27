
import torch
from torch import nn


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = False,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate tokens autoregressively from a language model.

    At each generation step, the model predicts the next token using
    the available context. The predicted token is appended to the
    sequence and used as context for the following step.

    Args:
        model: Language model that returns logits of shape (B, L, V).
        input_ids: Initial token IDs of shape (B, L).
        max_new_tokens: Maximum number of tokens to generate.
        max_seq_len: Maximum context length supported by the model.
        temperature: Sampling temperature. Higher values produce
            more diverse sampling distributions.
        top_k: If provided, restrict sampling to the top k logits.
        do_sample: If True, sample from the probability distribution.
            Otherwise, use greedy decoding.
        eos_token_id: Optional end-of-sequence token ID.

    Returns:
        Generated token IDs, including the original input sequence.

    Shape:
        - Input: (B, L)
        - Output: (B, L + generated_tokens)

    Example:
        >>> generated = generate(
        ...     model,
        ...     input_ids,
        ...     max_new_tokens=50,
        ...     max_seq_len=256,
        ...     do_sample=True,
        ...     top_k=50,
        ... )
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (B, L).")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative.")

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive.")

    if input_ids.size(1) == 0:
        raise ValueError("input_ids cannot be empty.")

    was_training = model.training
    model.eval()

    generated = input_ids

    try:
        for _ in range(max_new_tokens):

            # Keep only the most recent context tokens.
            context = generated[:, -max_seq_len:]

            # Get next-token logits.
            logits = model(context)

            next_token_logits = logits[:, -1, :]

            if do_sample:
                next_token_logits = next_token_logits / temperature

                if top_k is not None:
                    k = min(top_k, next_token_logits.size(-1))

                    values, _ = torch.topk(next_token_logits, k)

                    threshold = values[:, -1].unsqueeze(-1)

                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < threshold,
                        float("-inf"),
                    )

                probabilities = torch.softmax(
                    next_token_logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

            else:
                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1,
                    keepdim=True,
                )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            if eos_token_id is not None:
                if torch.all(next_token == eos_token_id):
                    break

    finally:
        model.train(was_training)

    return generated
