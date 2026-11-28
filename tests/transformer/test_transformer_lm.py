
import torch
import pytest

from transformer import TransformerLM


@pytest.fixture
def layer() -> TransformerLM:
    return TransformerLM(
        vocab_size=1000,
        num_layers=2,
        hidden_dim=256,
        num_heads=4,
        max_seq_len=256,
    )


def test_output_shape(layer: TransformerLM) -> None:
    """Test that the model returns logits with the expected shape."""
    input_ids = torch.randint(0, 1000, (2, 64))

    logits = layer(input_ids)

    assert logits.shape == (2, 64, 1000)


def test_causal_property(layer: TransformerLM) -> None:
    """Test that future tokens do not affect past predictions."""
    input_ids = torch.randint(0, 1000, (2, 64))
    modified_ids = input_ids.clone()

    cutoff = 32

    modified_ids[:, cutoff:] = torch.randint(
        0, 1000, (2, 64 - cutoff)
    )

    layer.eval()

    with torch.no_grad():
        logits_original = layer(input_ids)
        logits_modified = layer(modified_ids)

    assert torch.allclose(
        logits_original[:, :cutoff, :],
        logits_modified[:, :cutoff, :],
        atol=1e-5,
    )


def test_gradient_flow(layer: TransformerLM) -> None:
    """Test that gradients propagate through the model."""
    input_ids = torch.randint(0, 1000, (2, 64))
    targets = torch.randint(0, 1000, (2, 64))

    logits = layer(input_ids)

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 1000),
        targets.reshape(-1),
    )

    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in layer.parameters()
        if parameter.requires_grad
    )


def test_numerical_stability(layer: TransformerLM) -> None:
    """Test that model outputs contain only finite values."""
    input_ids = torch.randint(0, 1000, (2, 64))

    layer.eval()

    with torch.no_grad():
        logits = layer(input_ids)

    assert torch.isfinite(logits).all()


def test_short_sequence(layer: TransformerLM) -> None:
    """Test that the model handles short input sequences."""
    input_ids = torch.randint(0, 1000, (2, 4))

    logits = layer(input_ids)

    assert logits.shape == (2, 4, 1000)


def test_max_sequence_length(layer: TransformerLM) -> None:
    """Test that the model handles its maximum sequence length."""
    input_ids = torch.randint(0, 1000, (2, 256))

    logits = layer(input_ids)

    assert logits.shape == (2, 256, 1000)
