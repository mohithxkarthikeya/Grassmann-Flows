
import torch
from torch import nn

from common.embeddings import TokenEmbedding, PositionalEmbedding
from common.language_model_head import LanguageModelHead
from ..blocks import TransformerBlock


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        num_layers: Number of Transformer blocks. Default: 6.
        hidden_dim: Dimension of hidden representations. Default: 256.
        num_heads: Number of attention heads. Default: 4.
        max_seq_len: Maximum supported sequence length. Default: 256.
        expansion_factor: Feed-forward expansion factor. Default: 4.
        dropout: Dropout probability. Default: 0.1.
        tie_weights: Whether to share token embedding and LM head
            weights. Default: False.

    Shape:
        - Input: (B, L)
        - Output: (B, L, V)

        B: Batch size.
        L: Sequence length.
        V: Vocabulary size.

    Attributes:
        token_embedding: Token embedding module.
        position_embedding: Positional embedding module.
        embedding_dropout: Dropout applied to input embeddings.
        blocks: Stack of Transformer blocks.
        final_norm: Final layer normalization.
        lm_head: Vocabulary projection layer.
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 6,
        hidden_dim: int = 256,
        num_heads: int = 4,
        max_seq_len: int = 256,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        tie_weights: bool = False,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
        )

        self.position_embedding = PositionalEmbedding(
            max_seq_len=max_seq_len,
            hidden_dim=hidden_dim,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.lm_head = LanguageModelHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            bias=False,
        )

        # Optionally share token embedding and output projection weights.
        if tie_weights:
            self.lm_head.projection.weight = (
                self.token_embedding.embedding.weight
            )

        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        r"""Generate vocabulary logits for each input token position.

        Args:
            input_ids: Input token IDs of shape (B, L).

        Returns:
            Vocabulary logits of shape (B, L, V).
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, L).")

        seq_len = input_ids.size(1)

        if seq_len <= 0 or seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length must be between 1 and "
                f"{self.max_seq_len}."
            )

        # Convert token IDs into hidden representations.
        x = self.token_embedding(input_ids)

        # Add positional information.
        x = x + self.position_embedding(seq_len)

        # Apply embedding dropout.
        x = self.embedding_dropout(x)

        # Process representations through Transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Normalize and project into vocabulary logits.
        x = self.final_norm(x)

        return self.lm_head(x)
