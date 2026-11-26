
import torch
from torch import nn


class TokenEmbedding(nn.Module):
    r"""Converts token IDs into dense vector representations.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        hidden_dim: Embedding dimension.

    Shape:
        - Input: (B, L)
        - Output: (B, L, H)

    Attributes:
        embedding: Learnable token embedding table.

    Example:
        >>> embedding = TokenEmbedding(30522, 256)
        >>> tokens = torch.randint(0, 30522, (8, 128))
        >>> output = embedding(tokens)
        >>> print(output.shape)
        torch.Size([8, 128, 256])
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs into embeddings."""

        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    r"""Provides learnable positional representations for sequences.

    Each sequence position receives a learnable vector that can be added
    to its corresponding token embedding.

    Args:
        max_seq_len: Maximum supported sequence length.
        hidden_dim: Embedding dimension.

    Shape:
        - Input: Integer sequence length.
        - Output: (1, L, H)

    Attributes:
        embedding: Learnable positional embedding table.

    Example:
        >>> position_embedding = PositionalEmbedding(256, 256)
        >>> positions = position_embedding(128)
        >>> print(positions.shape)
        torch.Size([1, 128, 256])
    """

    def __init__(
        self,
        max_seq_len: int,
        hidden_dim: int,
    ):
        super().__init__()

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=hidden_dim,
        )

    def forward(
        self,
        seq_len: int,
    ) -> torch.Tensor:
        """Return positional embeddings for a sequence."""

        if seq_len <= 0 or seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len must be between 1 and {self.max_seq_len}."
            )

        positions = torch.arange(
            seq_len,
            device=self.embedding.weight.device,
        )

        return self.embedding(positions).unsqueeze(0)