
import torch
from torch import nn

from ..blocks.grassmann_block import CausalGrassmannBlock
from common.embeddings import TokenEmbedding, PositionalEmbedding
from common.language_model_head import LanguageModelHead


class GrassmannLM(nn.Module):
    r"""Language model built by stacking Causal Grassmann blocks.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        num_layers: Number of Grassmann blocks.
        hidden_dim: Dimension of token representations.
        reduced_dim: Dimension of reduced Grassmann representations.
        max_seq_len: Maximum supported sequence length.
        expansion_factor: Feed-forward expansion factor.
        dropout: Dropout probability.
        window_schedule: One geometric offset per block.
        tie_weights: Whether to share token embedding and LM head weights.

    Shape:
        - Input: (B, L)
        - Output: (B, L, V)
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 6,
        hidden_dim: int = 256,
        reduced_dim: int = 32,
        max_seq_len: int = 256,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        window_schedule: list[int] | None = None,
        tie_weights: bool = False,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if window_schedule is None:
            schedules = {
                6: [1, 2, 4, 8, 12, 16],
                12: [1, 1, 2, 2, 4, 4, 8, 8, 12, 12, 16, 16],
            }

            if num_layers not in schedules:
                raise ValueError(
                    "Provide window_schedule for this number of layers."
                )

            window_schedule = schedules[num_layers]

        if len(window_schedule) != num_layers:
            raise ValueError(
                "window_schedule must contain one offset per layer."
            )

        if any(offset <= 0 for offset in window_schedule):
            raise ValueError("All window offsets must be positive.")

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
            CausalGrassmannBlock(
                hidden_dim=hidden_dim,
                reduced_dim=reduced_dim,
                offsets=[offset],
                expansion_factor=expansion_factor,
                dropout=dropout,
            )
            for offset in window_schedule
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.lm_head = LanguageModelHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            bias=False,
        )

        if tie_weights:
            self.lm_head.projection.weight = (
                self.token_embedding.embedding.weight
            )

        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        r"""Generate vocabulary logits for each input position."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, L).")

        seq_len = input_ids.size(1)

        if seq_len <= 0 or seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length must be between 1 and "
                f"{self.max_seq_len}."
            )

        token_features = self.token_embedding(input_ids)
        position_features = self.position_embedding(seq_len)

        x = token_features + position_features
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        return self.lm_head(x)
