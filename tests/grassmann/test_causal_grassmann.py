import pytest
import torch
from grassmann import CausalGrassmannMixing

@pytest.fixture
def layer():
    return CausalGrassmannMixing(
        reduced_dim=8,
        hidden_dim=32,
        offsets=[1, 2, 4],
        dropout=0.0,
    )


def test_output_shape(layer: CausalGrassmannMixing) -> None:
    """Test that ouput dimensions match input dimensions"""
    x = torch.randn(2, 16, 32)

    output = layer(x)

    assert output.shape == x.shape


def test_plucker_encoding(layer: CausalGrassmannMixing) -> None:
    """Test the shape and numeric validity of plucker endocdings"""
    u = torch.tensor([[1.0, 2.0, 3.0, 0, 0, 0, 0, 0]])
    v = torch.tensor([[4.0, 5.0, 6.0, 0, 0, 0, 0, 0]])

    output = layer.plucker_encoding(u, v)

    assert output.shape == (1, 28)
    assert torch.isfinite(output).all()


def test_causal_property(layer: CausalGrassmannMixing) -> None:
    """Test that future input changes do not affect past outputs."""
    layer.eval()

    x1 = torch.randn(2, 16, 32)
    x2 = x1.clone()

    # Change the current and future input tokens.
    cutoff = 8
    x2[:, cutoff:, :] = torch.randn_like(x2[:, cutoff:, :])

    y1 = layer(x1)
    y2 = layer(x2)

    # Earlier outputs must remain unchanged.
    torch.testing.assert_close(
        y1[:, :cutoff, :],
        y2[:, :cutoff, :],
    )


def test_short_sequence(layer: CausalGrassmannMixing) -> None:
    """Test that the layer handles sequences shorter than its offsets."""
    x = torch.randn(2, 1, 32)

    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_gradient_flow(layer: CausalGrassmannMixing) -> None:
    """Test that gradients propagate through the input and parameters."""
    x = torch.randn(2, 16, 32, requires_grad=True)

    output = layer(x)
    loss = output.square().mean()

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_invalid_offset():
    """Test that non-positive sequence offsets raise a ValueError."""
    layer = CausalGrassmannMixing(
        reduced_dim=8,
        hidden_dim=32,
        offsets=[0],
    )

    x = torch.randn(2, 16, 32)

    with pytest.raises(ValueError):
        layer(x)


def test_numerical_stability(layer: CausalGrassmannMixing) -> None:
    """Test that zero-valued inputs produce finite outputs."""
    x = torch.zeros(2, 16, 32)

    output = layer(x)

    assert torch.isfinite(output).all()
