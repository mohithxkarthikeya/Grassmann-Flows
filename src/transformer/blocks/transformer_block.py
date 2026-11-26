import torch
from torch import nn

from common.layers.feedforward import FeedForward
from ..layers import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    r"""Applies a standard decoder-only Transformer block.

    The block consists of two sublayers:

    1. Causal multi-head self-attention.
    2. Position-wise feed-forward network.

    Each sublayer uses pre-layer normalization and a residual connection.

    Args:
        hidden_dim: Dimension of input and output representations.
            Default: 256.
        num_heads: Number of attention heads. Default: 4.
        expansion_factor: Feed-forward expansion factor. Default: 4.
        dropout: Dropout probability used in attention and feed-forward
            layers. Default: 0.1.

    Shape:
        - Input: (B, L, H)
        - Output: (B, L, H)

        B: Batch size.
        L: Sequence length.
        H: Hidden dimension.

    Attributes:
        attention_norm: Layer normalization before self-attention.
        attention: Causal multi-head self-attention layer.
        feedforward_norm: Layer normalization before feed-forward.
        feedforward: Position-wise feed-forward network.

    Example:
        >>> block = TransformerBlock(
        ...     hidden_dim=256,
        ...     num_heads=4
        ... )
        >>> x = torch.randn(8, 128, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([8, 128, 256])
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim)

        self.attention = MultiHeadSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feedforward_norm = nn.LayerNorm(hidden_dim)

        self.feedforward = FeedForward(
            hidden_dim=hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply attention and feed-forward transformations.

        Args:
            x: Input sequence of hidden representations.

        Returns:
            Transformed hidden representations.
        """

        # Causal self-attention with residual connection.
        x = x + self.attention(self.attention_norm(x))

        # Feed-forward network with residual connection.
        x = x + self.feedforward(self.feedforward_norm(x))

        return x
