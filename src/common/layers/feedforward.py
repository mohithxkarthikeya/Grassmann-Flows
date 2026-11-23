
import torch
from torch import nn


class FeedForward(nn.Module):
    r"""Position-wise feed-forward network.

    Applies two linear transformations with a GELU activation between them.
    The hidden dimension is expanded and then projected back to its original
    size.

    Args:
        hidden_dim: Input and output feature dimension.
        expansion_factor: Factor used to expand the hidden dimension.
        dropout: Dropout probability.

    Shape:
        - Input: (B, L, H)
        - Output: (B, L, H)

        B: Batch size.
        L: Sequence length.
        H: Hidden dimension.

    Attributes:
        network: Sequential feed-forward layers.

    Example:
        >>> layer = FeedForward(hidden_dim=256)
        >>> x = torch.randn(8, 128, 256)
        >>> output = layer(x)
        >>> print(output.shape)
        torch.Size([8, 128, 256])
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        ff_dim = hidden_dim * expansion_factor

        self.network = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transformation."""

        return self.network(x)
