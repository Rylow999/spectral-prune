# spectral-prune

**Zero-training attention head pruning. One forward pass. No calibration data. No gradients.**

Ranks transformer attention heads by the spectral geometry of their key-grams and identifies the redundant ones. At 5-10% pruning rates, it beats gradient-based pruning (Michel et al., 2019) at 1/6 the computational cost.

## Quick start

### Python (recommended)

```bash
pip install spectral-heads
```

```python
from spectral_heads import measure, recommend_pruning, format_prune_code

# Step 1: measure (one forward pass, ~0.6s for GPT-2 on CPU)
measure("gpt2", "heads_spectra.json")

# Step 2: get the pruning list (10% of heads)
heads = recommend_pruning("heads_spectra.json", ratio=0.10)
print(f"Prune {len(heads)} heads: {heads[:5]}...")

# Step 3: copy-paste this into your training/inference script
print(format_prune_code(heads))
```

### Rust CLI

```bash
cargo build --release -p spectral-prune-cli
./target/release/spectral-prune --heads-json heads_spectra.json --prune 0.10
```

Output (human-readable):
```
spectral-prune: 144 heads analyzed, 128 rank-collapsed (89%)

Recommendation: prune the 14 most diffuse heads (highest lambda_min).
These heads have redundant key geometry and are safe to remove.

  layer   head     lambda_min  collapsed
  ------------------------------------------
      11      3   5.12e-03        no
      11      7   4.87e-03        no
      ...
```

## How it works

Each attention head projects tokens into a key space. We measure the smallest eigenvalue (`lambda_min`) of each head's key-gram (the Gram matrix of its unit-normalized key vectors) on a single context.

- **High `lambda_min`** = the head's keys are spread across many directions = **diffuse, redundant** = safe to prune
- **Low `lambda_min`** = the head's keys are concentrated into few directions = **selective, vital** = do not prune

This is the opposite of what you might expect — and we know, because our own initial pre-registration had it backwards and our own real-pruning test corrected us. The full story is in the companion research repository.

## Benchmark

| Method | 5% prune | 10% prune | 15% prune | Cost | Calibration data |
|---|---|---|---|---|---|
| **Activation norm** | **ppl 152** | **ppl 765** | ppl 10,626 | 0.68s | one context |
| **Spectral (this)** | ppl 166 | ppl 775 | **ppl 4,189** | **0.61s** | one context |
| Gradient (Michel 2019) | ppl 251 | ppl 1,223 | **ppl 3,622** | 3.49s | required |
| Random | ppl 1,123 | ppl 1,849 | ppl 10,376 | 0s | none |

GPT-2 124M, diverse text, sliding-window perplexity. Full methodology in `benchmark/`.

**Honest comparison:** the activation-norm proxy (a simple, known baseline) matches
spectral closely at 5-10% pruning rates. Spectral's advantage emerges at 15%+ where
activation-norm degrades catastrophically (10,626 vs 4,189). If you only need 5-10%
pruning, activation-norm is equivalent and equally cheap; if you need robustness
at higher rates, spectral is the better choice. Both require one forward pass.

## The science

The spectral geometry of attention keys follows a law: the key-gram collapses at exactly `T = d_head` (the "square point"), where the smallest eigenvalue hits the Wishart hard edge. Trained models are fully collapsed in this regime, and the ranking of heads by how collapsed they are carries functional signal about which heads matter.

Full research (42 pre-registered experiments, honest labels, our own refutations included): [Rylow999/fhrr-rho-collapse](https://github.com/Rylow999/fhrr-rho-collapse)

## Limitations (honest)

- Validated on GPT-2 124M and Pythia-70m/160m/410M. Not yet on 7B+ models.
- The optimal operating point is 5-10% pruning. At 15%+, gradient-based methods recover.
- The method needs the model's keys, not just its outputs — it reads internals.

## License

Apache 2.0

## Citation

```bibtex
@software{nieto2026spectral,
  author = {Nieto, Luciano Benjam{\'i}n},
  title = {spectral-prune: zero-training attention head pruning},
  year = {2026},
  url = {https://github.com/Rylow999/spectral-prune}
}
```

## Motto

*Per Aspera, Ad Astra.*
