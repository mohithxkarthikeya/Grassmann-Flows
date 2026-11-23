import pytest
import torch
from grassmann import CausalGrassmannBlock



@pytest.fixture
def block() -> CausalGrassmannBlock:
    """Create a Grassmann block for testing."""
    return CausalGrassmannBlock(
        hidden_dim=32,
        reduced_dim=8,
        offsets=[1, 2, 4],
        dropout=0.0,
    )


def test_output_shape(block: CausalGrassmannBlock) -> None:
    """Test that the block preserves input dimensions."""
    x = torch.randn(2, 16, 32)

    output = block(x)

    assert output.shape == x.shape


def test_gradient_flow(block: CausalGrassmannBlock) -> None:
    """Test that gradients propagate through the block."""
    x = torch.randn(2, 16, 32, requires_grad=True)

    output = block(x)
    loss = output.square().mean()

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_causal_property(block: CausalGrassmannBlock) -> None:
    """Test that future inputs do not affect past outputs."""
    block.eval()

    x1 = torch.randn(2, 16, 32)
    x2 = x1.clone()

    x2[:, 8:, :] = torch.randn_like(x2[:, 8:, :])

    y1 = block(x1)
    y2 = block(x2)

    torch.testing.assert_close(
        y1[:, :8, :],
        y2[:, :8, :],
    )


def test_numerical_stability(block: CausalGrassmannBlock) -> None:
    """Test that the block produces finite outputs."""
    x = torch.zeros(2, 16, 32)

    output = block(x)

    assert torch.isfinite(output).all()
