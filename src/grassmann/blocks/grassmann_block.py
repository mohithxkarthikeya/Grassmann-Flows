
import torch
from torch import nn

from ..layers.grassmann_mixing import CausalGrassmannMixing
from common.layers.feedforward import FeedForward


class CausalGrassmannBlock(nn.Module):
    """Grassmann mixing block with a residual feed-forward network.

    Args:
        hidden_dim: Dimension of input and output representations.
            Default: 256.
        reduced_dim: Dimension of reduced Grassmann representations.
            Default: 32.
        offsets: Sequence offsets used by the Grassmann mixing layer.
        expansion_factor: Feed-forward expansion factor.
            Default: 4.
        dropout: Dropout probability.
            Default: 0.1.

    Shape:
        - Input: (B, L, H)
        - Output: (B, L, H)

        B: Batch size.
        L: Sequence length.
        H: Hidden dimension.

    Attributes:
        mixing: Causal Grassmann mixing layer.
        feedforward: Position-wise feed-forward network.
        norm: Layer normalization after the feed-forward residual.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        reduced_dim: int = 32,
        offsets: list[int] | None = None,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mixing = CausalGrassmannMixing(
            hidden_dim=hidden_dim,
            reduced_dim=reduced_dim,
            offsets=offsets,
            dropout=dropout,
        )

        self.feedforward = FeedForward(
            hidden_dim=hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Grassmann mixing.

        Args:
            x: Input hidden representations of shape (B, L, H).
        """
        x = self.mixing(x)

        out = self.feedforward(x)

        x = self.norm(x + out)

        return x
