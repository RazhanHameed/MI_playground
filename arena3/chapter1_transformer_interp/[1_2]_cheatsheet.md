
# TransformerLens & Induction Heads Cheat Sheet

## Core TransformerLens Features

### Loading Models
```python
# Load a pre-trained model
model = HookedTransformer.from_pretrained("gpt2-small")

# Load with custom config
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    attention_dir="causal",
    n_ctx=2048,
    ...
)
model = HookedTransformer(cfg)
```

### Basic Model Operations
```python
# Run model forward
logits = model(text, return_type="logits")
loss = model(text, return_type="loss") 
logits, cache = model.run_with_cache(text)

# Tokenization
tokens = model.to_tokens(text)
str_tokens = model.to_str_tokens(text) 
text = model.to_string(tokens)
```

### Accessing Model Weights
```python
# Direct weight access
W_E = model.W_E  # Embedding
W_U = model.W_U  # Unembedding
W_pos = model.W_pos  # Positional embedding
W_Q = model.W_Q  # Query weights [layer, head]
W_K = model.W_K  # Key weights
W_V = model.W_V  # Value weights
W_O = model.W_O  # Output weights
```

## Working with Activation Cache

### Key Cache Access Patterns
```python
# Access patterns - both equivalent:
attn_pattern = cache["pattern", 0]  # Layer 0 attention
attn_pattern = cache["blocks.0.attn.hook_pattern"]

# Common activations:
embed = cache["embed"]  # Embeddings
q = cache["q", layer]  # Queries
k = cache["k", layer]  # Keys
v = cache["v", layer]  # Values
pattern = cache["pattern", layer]  # Attention patterns
result = cache["result", layer]  # Head outputs
```

## Working with Hooks

### Basic Hook Structure
```python
def hook_function(activation, hook):
    # Modify or access activation
    return modified_activation  # Optional

# Apply hook
model.run_with_hooks(
    tokens,
    fwd_hooks=[
        ("blocks.0.attn.hook_pattern", hook_function)
    ]
)
```

### Common Hook Patterns
```python
# Access activation without modifying
def access_hook(act, hook):
    # Store or process activation
    pass

# Ablation hook
def ablation_hook(act, hook):
    act[:, :, head_idx] = 0.0
    return act

# Name filter hook
model.run_with_hooks(
    tokens,
    fwd_hooks=[
        (lambda name: name.endswith("pattern"), hook_function)
    ]
)
```

## Induction Heads

### Key Characteristics
- Formed by composition of:
  1. Previous token head (layer 0)
  2. Induction head (layer 1)
- Purpose: Learn & predict repeated sequences
- Identified by diagonal stripe pattern in attention

### Detection Methods
```python
# Attention pattern detector
def induction_score(pattern):
    return pattern.diagonal(-seq_len+1).mean()

# Direct logit attribution
logit_attr = W_E @ W_OV @ W_U

# Composition scores
def comp_score(W_A, W_B):
    return (W_A @ W_B).norm() / (W_A.norm() * W_B.norm())
```

## Visualization Tools

### CircuitsVis
```python
# Attention patterns
cv.attention.attention_patterns(
    tokens=str_tokens,
    attention=attention_pattern,
    attention_head_names=[f"L{layer}H{head}" for head in range(n_heads)]
)

# Attention heads
cv.attention.attention_heads(
    tokens=str_tokens, 
    attention=attention_pattern
)
```

## Common Analysis Patterns

### Basic Matrix Operations
```python
# OV Circuit
W_OV = W_V @ W_O  # Shape: [d_model, d_model]

# QK Circuit
W_QK = W_Q @ W_K.T  # Shape: [d_model, d_model]

# Full Circuit
full_OV = W_E @ W_OV @ W_U  # Shape: [d_vocab, d_vocab]
```

