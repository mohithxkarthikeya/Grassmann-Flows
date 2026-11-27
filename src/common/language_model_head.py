
import torch
from torch import nn


class LanguageModelHead(nn.Module):
    r"""Projects hidden representations into vocabulary logits.

    Applies a linear transformation to produce a score for every
    vocabulary token at each sequence position.

    Args:
        hidden_dim: Dimension of the input representations.
        vocab_size: Number of tokens in the vocabulary.
        bias: Whether to include a learnable bias. Default: False.

    Shape:
        - Input: (B, L, H)
        - Output: (B, L, V)

        B: Batch size.
        L: Sequence length.
        H: Hidden dimension.
        V: Vocabulary size.

    Attributes:
        projection: Linear vocabulary projection.

    Example:
        >>> head = LanguageModelHead(256, 30522)
        >>> x = torch.randn(8, 128, 256)
        >>> logits = head(x)
        >>> print(logits.shape)
        torch.Size([8, 128, 30522])
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        bias: bool = False,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        self.projection = nn.Linear(
            hidden_dim,
            vocab_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert hidden representations into vocabulary logits."""

        return self.projection(x)
