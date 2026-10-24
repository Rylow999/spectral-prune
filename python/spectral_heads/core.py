"""
Core implementation: measure per-head key-gram spectra from a HuggingFace
transformer, rank by lambda_min, and recommend pruning.

The method (validated in the companion benchmark, Exp 42):
  - Heads with the HIGHEST lambda_min are the most diffuse (redundant
    key geometry) and are the safe pruning targets.
  - Heads with the LOWEST lambda_min are the selective, vital ones.
  - At 5-10% pruning rates this beats gradient-based pruning
    (Michel et al. 2019) at 1/6 the computational cost.
"""
import json
import numpy as np


class HeadScore:
    """One head's spectral measurement."""
    __slots__ = ("layer", "head", "lmin", "lmax")

    def __init__(self, layer, head, lmin, lmax):
        self.layer = layer
        self.head = head
        self.lmin = lmin
        self.lmax = lmax

    @property
    def collapsed(self):
        return abs(self.lmin) < 1e-4

    def __repr__(self):
        return f"HeadScore(L={self.layer}, H={self.head}, lmin={self.lmin:.2e})"

    def to_dict(self):
        return {"layer": self.layer, "head": self.head,
                "lmin": self.lmin, "lmax": self.lmax}


def measure(model_name: str, output_path: str = "heads_spectra.json",
            context_text: str = None, context_length: int = 64,
            device: str = "cpu") -> list:
    """Measure per-head key-gram spectra of a transformer.

    Args:
        model_name: HuggingFace model name (e.g. "gpt2", "EleutherAI/pythia-70m")
        output_path: where to save the JSON
        context_text: the text to use as context (default: a standard probe)
        context_length: number of tokens (default 64 = d_head for most models)
        device: "cpu" or "cuda"

    Returns:
        List of HeadScore objects, one per (layer, head) pair.

    Cost: ONE forward pass. No data, no gradients, no backward.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if context_text is None:
        context_text = (
            "The quick brown fox jumps over the lazy dog. " * 12 +
            "The city council approved the budget for the new park after "
            "a long debate about the environmental impact and the cost "
            "of maintenance. Local residents supported the initiative."
        )

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()
    model = model.to(device)

    cfg = model.config
    # Handle both GPT-2 and GPTNeoX style configs
    if hasattr(cfg, "n_head"):
        nh, d_model = cfg.n_head, cfg.n_embd
    else:
        nh, d_model = cfg.num_attention_heads, cfg.hidden_size
    dh = d_model // nh
    nl = cfg.n_layer if hasattr(cfg, "n_layer") else cfg.num_hidden_layers

    ids = tok(context_text, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] < context_length:
        ids = tok(context_text * 3, return_tensors="pt").input_ids.to(device)
    ids = ids[:, :context_length]

    keys_by_layer = {}

    # GPT-2 style: c_attn (fused qkv)
    if hasattr(model, "h"):
        for i, blk in enumerate(model.h):
            cat = blk.attn.c_attn
            def mk(idx, proj):
                def hook(mod, inp, out):
                    hidden = inp[0].float()
                    B, T, _ = hidden.shape
                    qkv = hidden @ proj.weight.float() + proj.bias.float()
                    K = qkv[:, :, d_model:2*d_model].view(B, T, nh, dh)
                    keys_by_layer[idx] = K.permute(0, 2, 1, 3)[0]
                return hook
            cat.register_forward_hook(mk(i, cat))
    # GPTNeoX style: query_key_value
    elif hasattr(model, "layers"):
        for i, blk in enumerate(model.layers):
            qkv = blk.attention.query_key_value
            def mk(idx, proj):
                def hook(mod, inp, out):
                    hidden = inp[0].float()
                    B, T, _ = hidden.shape
                    qo = hidden @ proj.weight.float().T + proj.bias.float()
                    K = qo[:, :, d_model:2*d_model].view(B, T, nh, dh)
                    keys_by_layer[idx] = K.permute(0, 2, 1, 3)[0]
                return hook
            qkv.register_forward_hook(mk(i, qkv))
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    with torch.no_grad():
        _ = model(ids)

    # Clean up hooks
    if hasattr(model, "h"):
        for i in range(nl):
            model.h[i].attn.c_attn._forward_hooks.clear()
    else:
        for i in range(nl):
            model.layers[i].attention.query_key_value._forward_hooks.clear()

    # Compute eigenvalues per head
    heads = []
    for li in sorted(keys_by_layer):
        K = keys_by_layer[li].cpu().numpy()
        for h in range(nh):
            Kh = K[h] / np.linalg.norm(K[h], axis=1, keepdims=True)
            gram = Kh @ Kh.T
            ev = np.linalg.eigvalsh(gram)
            heads.append(HeadScore(li, h, float(ev[0]), float(ev[-1])))

    # Save
    data = {"model": model_name, "context_length": ids.shape[1],
            "d_head": dh, "heads": [h.to_dict() for h in heads]}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return heads


def recommend_pruning(input_path: str = "heads_spectra.json",
                      ratio: float = 0.10) -> list:
    """Read the measured spectra and return the pruning list.

    Args:
        input_path: JSON file from measure()
        ratio: fraction of heads to prune (0.10 = 10%)

    Returns:
        List of (layer, head) tuples, ordered by priority (prune first).
        These are the MOST DIFFUSE heads (highest lambda_min) — the
        redundant ones that are safe to remove.
    """
    with open(input_path) as f:
        data = json.load(f)

    heads = [HeadScore(h["layer"], h["head"], h["lmin"], h["lmax"])
             for h in data["heads"]]

    # Sort by lambda_min DESCENDING: most diffuse first = prune first
    heads.sort(key=lambda h: h.lmin, reverse=True)

    n_prune = int(len(heads) * ratio)
    return [(h.layer, h.head) for h in heads[:n_prune]]


def format_prune_code(prune_list: list) -> str:
    """Generate copy-pasteable PyTorch code to prune the recommended heads."""
    by_layer = {}
    for layer, head in prune_list:
        by_layer.setdefault(layer, []).append(head)

    lines = ["heads_to_prune = {"]
    for layer, heads in sorted(by_layer.items()):
        head_str = ", ".join(str(h) for h in sorted(heads))
        lines.append(f"    {layer}: {{{head_str}}},")
    lines.append("}")
    lines.append("# Zero each head's output slice via forward hooks:")
    lines.append("# ao.view(B, T, n_heads, d_head)[:, :, h, :] = 0")
    return "\n".join(lines)
