
import torch
from torch import nn
from torch.nn.functional import normalize


class CausalGrassmannMixing(nn.Module):
    r"""Applies Grassmannian geometric mixing to sequential hidden representations.

    The layer projects hidden states into a lower-dimensional space,
    constructs Plücker coordinates from token pairs at multiple sequence
    offsets, and projects the resulting geometric features back into
    the model dimension for gated fusion.

    Args:
        reduced_dim: Dimension of the reduced token representations.
            Default: 32.
        hidden_dim: Dimension of the input and output hidden representations.
            Default: 256.
        offsets: Sequence offsets used for geometric pairing.
            Default: [1, 2, 4, 8, 16, 32].
        dropout: Dropout probability applied after layer normalization.
            Default: 0.1.
        norm_eps: Epsilon for Plücker coordinate normalization.
            Default: 1e-6.

    Shape:
        - Input: :math:`(B, L, H)` where 
            :math:`B` is the batch size,
            :math:`L` is the sequence length, and
            :math:`H = \text{hidden\_dim}`.
        - Output: :math:`(B, L, H)`, same shape as the input.

    Attributes:
        input_projection: Learnable linear projection of shape
            :math:`(\text{reduced\_dim}, \text{hidden\_dim})`.
        plucker_projection: Learnable linear projection of shape
            :math:`(\text{hidden\_dim}, \binom{\text{reduced\_dim}}{2})`.
        mixing_gate: Learnable linear projection of shape
            :math:`(\text{hidden\_dim}, 2\text{hidden\_dim})`.
        norm: Layer normalization module.
        dropout: Dropout module.
        plucker_indices: Precomputed index pairs for Plücker coordinates.

    Examples::

        >>> layer = CausalGrassmannMixing(
        ...     reduced_dim=32,
        ...     hidden_dim=256,
        ...     offsets=[1, 2, 4, 8]
        ... )
        >>> x = torch.randn(8, 128, 256)
        >>> output = layer(x)
        >>> print(output.shape)
        torch.Size([8, 128, 256])
    """

    def __init__(
        self,
        reduced_dim: int = 32,
        hidden_dim: int = 256,
        offsets: list[int] | None = None,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        if reduced_dim < 2:
            raise ValueError("reduced_dim must be at least 2")

        self.reduced_dim = reduced_dim
        self.hidden_dim = hidden_dim
        self.offsets = (
            [1, 2, 4, 8, 16, 32]
            if offsets is None else offsets
        )
        self.norm_eps = norm_eps

        # 1. Linear reduction: hidden_dim -> reduced_dim
        self.input_projection = nn.Linear(hidden_dim, reduced_dim)

        # 2. Plücker projection: C(reduced_dim, 2) -> hidden_dim
        self.plucker_projection = nn.Linear(
            reduced_dim * (reduced_dim - 1) // 2, hidden_dim
        )

        # 3. Gated fusion: 2 * hidden_dim -> hidden_dim
        self.mixing_gate = nn.Linear(2 * hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Precompute coordinate index pairs (i, j), where i < j.
        self.register_buffer(
            "plucker_indices",
            torch.triu_indices(reduced_dim, reduced_dim, offset=1)
        )

    def plucker_encoding(
        self,
        u: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        r"""Computes normalized Plücker coordinates.

        Args:
            u: Reduced representation of the current token.
            v: Reduced representation of the future token.

        Shape:
            - u: :math:`(*, r)` where :math:`r = \text{reduced\_dim}`.
            - v: :math:`(*, r)`, same shape as u.
            - Output: :math:`(*, \binom{r}{2})`.
        """

        i, j = self.plucker_indices

        # Compute pairwise Plücker coordinates.
        p = u[..., i] * v[..., j] - u[..., j] * v[..., i]

        return normalize(p, p=2, dim=-1, eps=self.norm_eps)

    def geometric_aggregation(
        self,
        reduced_states: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregates geometric features across sequence offsets.

        Args:
            reduced_states: Sequence of reduced token representations.
        """
        _, L, _ = reduced_states.shape

        geometry_sum = reduced_states.new_zeros(
            *reduced_states.shape[:2], self.hidden_dim
        )

        valid_offset_count = reduced_states.new_zeros(
            *reduced_states.shape[:2], 1
        )

        for offset in self.offsets:

            if offset <= 0:
                raise ValueError("Offsets must be positive")

            if offset >= L:
                continue

            # Pair current token representations with future positions.
            u = reduced_states[:, :-offset, :]
            v = reduced_states[:, offset:, :]

            plucker_coords = self.plucker_encoding(u, v)
            projected = self.plucker_projection(plucker_coords)

            geometry_sum[:, offset:, :] += projected
            valid_offset_count[:, offset:, :] += 1

        return geometry_sum / valid_offset_count.clamp(min=1)

    def gated_fusion(
        self,
        h: torch.Tensor,
        geometry_features: torch.Tensor
    ) -> torch.Tensor:
        """Combines hidden representations using gated fusion.

        Args:
            h: Original hidden representations.
            geometry_features: Aggregated geometric features.
        """

        gate_input = torch.cat((h, geometry_features), dim=-1)

        hidden_gate = torch.sigmoid(self.mixing_gate(gate_input))

        mixed_states = hidden_gate * h + (1 - hidden_gate) * geometry_features

        return self.dropout(self.norm(mixed_states))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Applies Grassmann mixing."""

        reduced_states = self.input_projection(h)
        geometry_features = self.geometric_aggregation(reduced_states)

        return self.gated_fusion(h, geometry_features)
