"""Checks of the connection modules: exact agreement with each paper's Algorithm 2, exact
equality with a plain pre-norm network at init, gradients, bf16 streams, and layer readouts.
Run with `pytest tests`."""

import pytest
import torch

from residual_streams import Encoder, FracConnection, HyperConnection
from paper_reference import PaperFracConnection, PaperHyperConnection

torch.set_num_threads(2)


def perturb(module, scale=0.3):
    # move every parameter off its init so the checks are not trivial
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))
    return module


@pytest.mark.parametrize("dynamic", [True, False])
def test_hyper_connection_matches_paper(dynamic):
    torch.manual_seed(0)
    ours = perturb(HyperConnection(16, 4, layer_id=5, dynamic=dynamic).double())
    ref = PaperHyperConnection(16, 4, 5, dynamic).double()
    ref.load_state_dict(ours.state_dict())
    h = torch.randn(2, 7, 4, 16, dtype=torch.double)
    h_o = torch.randn(2, 7, 16, dtype=torch.double)
    x, alpha, beta = ours.width_connection(h)
    mix_h, ref_beta = ref.width_connection(h)
    torch.testing.assert_close(x, ref.branch_input(mix_h), rtol=0, atol=1e-12)
    torch.testing.assert_close(
        ours.depth_connection(h, h_o, alpha, beta), ref.depth_connection(mix_h, h_o, ref_beta), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("dynamic", [True, False])
def test_frac_connection_matches_paper(dynamic):
    torch.manual_seed(0)
    ours = perturb(FracConnection(48, 4, dynamic=dynamic).double())
    ref = PaperFracConnection(48, 4, dynamic).double()
    ref.load_state_dict(ours.state_dict())
    h = torch.randn(2, 7, 48, dtype=torch.double)
    h_o = torch.randn(2, 7, 48, dtype=torch.double)
    x, alpha, beta = ours.width_connection(h)
    mix_h, ref_beta = ref.width_connection(h)
    torch.testing.assert_close(x, ref.branch_input(mix_h), rtol=0, atol=1e-12)
    torch.testing.assert_close(
        ours.depth_connection(h, h_o, alpha, beta), ref.depth_connection(mix_h, h_o, ref_beta), rtol=0, atol=1e-12
    )


def residual_stream(encoder, x):
    """The encoder up to its final norm, streams averaged (hyper-connections)."""
    if encoder.rate:
        x = x[..., None, :].expand(*x.shape[:-1], encoder.rate, x.shape[-1])
    for block in encoder.blocks:
        x = block(x)
    return x.mean(dim=-2) if encoder.rate else x


@pytest.mark.parametrize("connections", ["hyper", "frac"])
@pytest.mark.parametrize("dynamic", [True, False])
def test_init_equals_plain_prenorm(connections, dynamic):
    # hyper eq. 14 and frac eq. 15: at init the network is exactly the plain pre-norm one
    torch.manual_seed(0)
    plain = Encoder(48, 3, 4).double()
    conn = Encoder(48, 3, 4, connections=connections, dynamic=dynamic).double()
    missing = conn.load_state_dict(plain.state_dict(), strict=False)
    assert not missing.unexpected_keys and all(".conn_" in k for k in missing.missing_keys)
    x = torch.randn(2, 9, 48, dtype=torch.double)
    assert torch.equal(residual_stream(conn, x), residual_stream(plain, x))


@pytest.mark.parametrize("connections", ["hyper", "frac"])
def test_every_parameter_gets_a_gradient(connections):
    torch.manual_seed(0)
    encoder = Encoder(48, 3, 4, connections=connections)
    encoder(torch.randn(2, 9, 48)).square().mean().backward()
    assert [n for n, p in encoder.named_parameters() if p.grad is None] == []


@pytest.mark.parametrize("connections", ["hyper", "frac"])
def test_last_layer_readout_equals_output(connections):
    torch.manual_seed(0)
    encoder = perturb(Encoder(48, 3, 4, connections=connections), 0.05)
    x = torch.randn(2, 9, 48)
    assert torch.equal(encoder.forward_layers(x, (3,))[0], encoder(x))


@pytest.mark.parametrize("connections", ["hyper", "frac"])
def test_bf16_streams(connections):
    torch.manual_seed(0)
    fp32 = perturb(Encoder(48, 3, 4, connections=connections), 0.05)
    bf16 = Encoder(48, 3, 4, connections=connections, stream_dtype=torch.bfloat16)
    bf16.load_state_dict(fp32.state_dict())
    x = torch.randn(2, 9, 48)
    h = x[..., None, :].expand(2, 9, 4, 48) if bf16.rate else x
    assert bf16.blocks[0](h).dtype == torch.bfloat16
    out32, out16 = fp32(x), bf16(x)
    assert out16.dtype == torch.float32
    assert ((out16 - out32).norm() / out32.norm()).item() < 1e-2
