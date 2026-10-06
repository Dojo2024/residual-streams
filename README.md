# residual-streams

Small, readable PyTorch implementations of learned residual connections, written to be read next
to the papers, with tests that check them against each paper's own pseudo-code. Experiments with
them are in [RESULTS.md](RESULTS.md).

| Method | Paper | What it changes |
|---|---|---|
| Hyper-connections (HC) | Zhu et al., *Hyper-Connections*, ICLR 2025, [arXiv 2409.19606](https://arxiv.org/abs/2409.19606) | Keeps `rate` copies (streams) of each token and learns how each block reads, writes and mixes them |
| Frac-connections (FC) | Zhu et al., *Frac-Connections: Fractional Extension of Hyper-Connections*, [arXiv 2503.14125](https://arxiv.org/abs/2503.14125) | Splits the hidden state into `rate` fractions instead of copying it, so no extra memory |
| mHC, mHC-lite | Xie et al., *mHC: Manifold-Constrained Hyper-Connections*, [arXiv 2512.24880](https://arxiv.org/abs/2512.24880); Yang et al., *mHC-lite*, [arXiv 2601.05732](https://arxiv.org/abs/2601.05732) | Constrains the stream mixing to doubly stochastic matrices so it cannot amplify over depth (planned) |

## The idea in one paragraph

A transformer block normally adds its output back to one running vector per token: `x = x + f(norm(x))`.
Hyper-connections replace that with `rate` vectors per token (the hyper hidden matrix `H`) and three
small learned matrices per branch:
- `A_m` mixes the streams into the branch's input;
- `A_r` mixes the streams with each other;
- `B` writes the branch output back into each stream.

So each block computes `H' = A_r^T H + B^T f(A_m^T H)`. The dynamic version adds input-dependent terms
`tanh(norm(H) W) * s`. The init makes the network exactly a plain pre-norm one at step 0, and learning
can then move it towards other arrangements (the paper's "sequential-parallel duality").
Frac-connections apply the same mixing to `rate` slices of one vector instead of `rate` copies.

## Layout

```
residual_streams/connections.py   HyperConnection, FracConnection (names follow each paper's Algorithm 2)
residual_streams/model.py         a minimal pre-norm encoder with connections="plain" | "hyper" | "frac"
tests/paper_reference.py          the papers' Algorithm 2, transcribed, as references
tests/test_connections.py         checks: exact match to the papers, exact plain network at init, grads, bf16
RESULTS.md                        experiments in a 3D MRI masked autoencoder
```

## Use

```python
from residual_streams import Encoder, HyperConnection

encoder = Encoder(dim=384, depth=12, heads=6, connections="hyper", rate=4)  # or "frac", "plain"
y = encoder(x)                                   # x: [batch, tokens, 384]
features = encoder.forward_layers(x, (6, 9, 12))  # per-layer readouts; HC layers = sum of streams
```

To add a connection to your own block, wrap each residual branch:

```python
x, alpha, beta = conn.width_connection(h)        # branch input
h = conn.depth_connection(h, branch(x), alpha, beta)
```

`dynamic=False` gives the static versions (SHC, SFC). `stream_dtype=torch.bfloat16` stores the streams
in bf16 between blocks, halving their memory traffic; the mixing itself stays in fp32. The papers' FC
code does the same.

## Notes on the implementation

- **Same math as the papers, less memory traffic.** The papers' `width_connection` returns `mix_h`
  (branch input and mixed streams stacked), which `depth_connection` reads back. Here
  `width_connection` returns only the branch input, and `depth_connection` forms the mixed streams
  itself, so `mix_h` is never stored. The tests check the result against the papers' code to 1e-12
  in float64.
- **The dynamic projections are one fp32 matmul.** `W_alpha` and `W_beta` are concatenated and run
  with autocast off, so mixing weights are not rounded to bf16.
- **Each branch counts as one layer for the HC init.** The attention of block `i` reads stream
  `2i mod rate` and its MLP `(2i + 1) mod rate`.
- **Optimizer.** In our runs the connection parameters were on AdamW, with no weight decay on the
  static matrices and scales, as in the papers.

## Tests

```
pip install torch pytest
pytest tests
```
