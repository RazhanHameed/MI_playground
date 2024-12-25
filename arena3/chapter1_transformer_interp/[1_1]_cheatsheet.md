# Transformer Implementation Cheatsheet

## Architecture Overview

- Input tokens → Embedding → Positional Embedding → Transformer Blocks → Final Layer Norm → Unembedding → Output logits
- Each transformer block: LayerNorm → Attention → LayerNorm → MLP (with residual connections)
- Uses causal attention (can only attend to past tokens)

## Key Components

### LayerNorm
```python
def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
    residual_mean = residual.mean(dim=-1, keepdim=True)
    residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()
    
    residual = (residual - residual_mean) / residual_std
    return residual * self.w + self.b
```

### Embedding
```python
def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
    # Simple lookup table operation
    return self.W_E[tokens]
```

### Positional Embedding
```python
def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
    batch, seq_len = tokens.shape
    return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)
```

### Attention
```python
def forward(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
    # 1. Calculate query, key and value vectors
    q = einops.einsum(normalized_resid_pre, self.W_Q,
        "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head") + self.b_Q
    k = einops.einsum(normalized_resid_pre, self.W_K,
        "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head") + self.b_K
    v = einops.einsum(normalized_resid_pre, self.W_V,
        "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head") + self.b_V

    # 2. Calculate attention scores
    attn_scores = einops.einsum(q, k,
        "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K")
    
    # 3. Scale, mask and apply softmax
    attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
    attn_pattern = attn_scores_masked.softmax(-1)

    # 4. Calculate output
    z = einops.einsum(v, attn_pattern,
        "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head")
    
    out = einops.einsum(z, self.W_O,
        "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model") + self.b_O

    return out
```

### MLP
```python
def forward(self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
    pre = einops.einsum(
        normalized_resid_mid, self.W_in,
        "batch position d_model, d_model d_mlp -> batch position d_mlp"
    ) + self.b_in
    post = gelu_new(pre)
    return einops.einsum(
        post, self.W_out,
        "batch position d_mlp, d_mlp d_model -> batch position d_model"
    ) + self.b_out
```

## Sampling Methods

### Basic Sampling
```python
def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
    return t.distributions.categorical.Categorical(logits=logits).sample().item()
```

### Temperature Scaling
```python
def apply_temperature(logits: Float[Tensor, "d_vocab"], temperature: float) -> Float[Tensor, "d_vocab"]:
    return logits / temperature
```

### Top-K Sampling
```python
def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
    top_k_logits, top_k_token_ids = logits.topk(k)
    sampled_token_idx = t.distributions.categorical.Categorical(logits=top_k_logits).sample()
    return top_k_token_ids[sampled_token_idx].item()
```

### Top-P (Nucleus) Sampling
```python
def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
    logits_sorted, indices = logits.sort(descending=True, stable=True)
    cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
    n_keep = max(t.searchsorted(cumul_probs, top_p, side="left").item() + 1, min_tokens_to_keep)
    keep_idx = indices[:n_keep]
    keep_logits = logits[keep_idx]
    sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
    return keep_idx[sample].item()
```

## Key Shape Conventions

- Input tokens: `[batch, seq_len]`
- Embedding output: `[batch, seq_len, d_model]`
- Attention scores: `[batch, n_heads, seq_len, seq_len]`
- MLP input/output: `[batch, seq_len, d_model]`
- Final logits: `[batch, seq_len, d_vocab]`

## Typical Hyperparameters

```python
@dataclass
class Config:
    d_model: int = 768        # Dimension of embeddings
    debug: bool = True        # Whether to print debug info
    layer_norm_eps: float = 1e-5  # Epsilon for layer norm
    d_vocab: int = 50257      # Vocabulary size
    init_range: float = 0.02  # Range for random init
    n_ctx: int = 1024        # Maximum context length
    d_head: int = 64         # Dimension of attention heads
    d_mlp: int = 3072       # Dimension of MLP layer (= 4 * d_model)
    n_heads: int = 12        # Number of attention heads
    n_layers: int = 12       # Number of transformer blocks
```

Note: For attention heads, usually `d_model = n_heads * d_head`. For MLP layers, usually `d_mlp = 4 * d_model`.