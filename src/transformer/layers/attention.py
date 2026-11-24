
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    r"""Applies causal multi-head self-attention.

    Args:
        hidden_dim: Dimension of input and output representations.
            Default: 256.
        num_heads: Number of parallel attention heads. 
            Default: 4.
        dropout: Dropout probability applied to attention weights
            and the output projection. 
            Default: 0.1.
        bias: Whether linear projections use learnable biases.
            Default: True.

    Shape:
        - Input: (B, L, H)
        - Output: (B, L, H)

        B: Batch size.
        L: Sequence length.
        H: Hidden dimension.

    Attributes:
        qkv_projection: Joint projection for queries, keys, and values.
        output_projection: Projection that combines attention heads.
        output_dropout: Dropout applied after the output projection.

    Example:
        >>> attention = MultiHeadSelfAttention(
        ...     hidden_dim=256,
        ...     num_heads=4
        ... )
        >>> x = torch.randn(8, 128, 256)
        >>> output = attention(x)
        >>> print(output.shape)
        torch.Size([8, 128, 256])
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.attention_dropout = dropout

        # Project input into queries, keys, and values.
        self.qkv_projection = nn.Linear(
            hidden_dim,
            3 * hidden_dim,
            bias=bias,
        )

        # Combine the outputs of all attention heads.
        self.output_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=bias,
        )

        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute causal multi-head self-attention.

        Args:
            x: Input sequence of hidden representations.

        Returns:
            Attention-transformed hidden representations.

        Shape:
            - Input: (B, L, H)
            - Output: (B, L, H)
        """

        batch_size, seq_len, _ = x.shape

        # Project and separate queries, keys, and values.
        qkv = self.qkv_projection(x)

        qkv = qkv.view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )

        # Rearrange into (3, B, heads, L, head_dim).
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(0)

        # Apply scaled dot-product attention with causal masking.
        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=(
                self.attention_dropout if self.training else 0.0
            ),
            is_causal=True,
        )

        # Merge attention heads.
        attention_output = (
            attention_output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )

        # Project and apply output dropout.
        output = self.output_projection(attention_output)
        output = self.output_dropout(output)

        return output
