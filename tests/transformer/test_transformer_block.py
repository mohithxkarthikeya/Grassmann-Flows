
import torch
import pytest

from transformer.blocks.transformer_block import TransformerBlock


@pytest.fixture
def layer() -> TransformerBlock:
    return TransformerBlock(
        hidden_dim=256,
        num_heads=4,
        expansion_factor=4,
        dropout=0.0,
    )


def test_output_shape(layer: TransformerBlock) -> None:
    """Test that the Transformer block preserves input dimensions."""
    x = torch.randn(2, 32, 256)

    output = layer(x)

    assert output.shape == x.shape


def test_gradient_flow(layer: TransformerBlock) -> None:
    """Test that gradients propagate through the Transformer block."""
    x = torch.randn(2, 32, 256, requires_grad=True)

    output = layer(x)
    loss = output.square().mean()

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert any(
        parameter.grad is not None
        for parameter in layer.parameters()
        if parameter.requires_grad
    )


def test_numerical_stability(layer: TransformerBlock) -> None:
    """Test that the block produces finite outputs."""
    x = torch.randn(2, 32, 256)

    layer.eval()

    with torch.no_grad():
        output = layer(x)

    assert torch.isfinite(output).all()


def test_batch_independence(layer: TransformerBlock) -> None:
    """Test that one batch sample does not affect another."""
    x = torch.randn(2, 32, 256)
    modified_x = x.clone()

    modified_x[1] = torch.randn_like(modified_x[1])

    layer.eval()

    with torch.no_grad():
        output = layer(x)
        modified_output = layer(modified_x)

    assert torch.allclose(
        output[0],
        modified_output[0],
        atol=1e-5,
    )
