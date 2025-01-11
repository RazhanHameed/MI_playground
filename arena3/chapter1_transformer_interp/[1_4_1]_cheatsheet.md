# TransformerLens IOI Analysis Cheatsheet

## 1️⃣ Model & Task Setup

**Indirect Object Identification (IOI) Task:**
- Complete sentences like "John and Mary went to the store, John gave a drink to __" with correct name (Mary)
- Metric: Logit difference between correct (IO) and incorrect (S) tokens
- Key positions: S1, S2 (duplicated subject), IO (indirect object), END (prediction position)

## 2️⃣ Direct Logit Attribution Tools

**Key Functions:**
```python
# Get accumulated residual stream up to layer N
cache.accumulated_resid(layer=N)

# Decompose residual stream into components
cache.decompose_resid(layer=N)

# Get per-head contributions
cache.stack_head_results(layer=N)

# Project onto logit difference direction
residual_stack_to_logit_diff(residual, cache, logit_diff_direction)
```

## 3️⃣ Activation Patching

**Core Concept:** Replace activations from corrupted run with clean run to measure importance

**Key Function Template:**
```python
def patch_activation(
    corrupted_activation: Tensor,
    hook: HookPoint,
    clean_cache: ActivationCache,
    pos: int
) -> Tensor:
    corrupted_activation[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_activation
```

## 4️⃣ Path Patching 

**Core Concept:** Trace causal paths between model components

**3-Step Algorithm:**
1. Cache activations from clean & corrupted runs
2. Run on clean input with sender patched from corrupted, other heads frozen
3. Use cached receiver activation from step 2 for final run

**Key Function Template:**
```python
def path_patch(
    sender: tuple[int, int],  # (layer, head)
    receiver_heads: list[tuple[int, int]], 
    receiver_input: Literal["k", "q", "v"]
) -> float:
    # 1. Cache activations
    # 2. Freeze all heads except sender
    # 3. Patch receiver input
    # 4. Return performance metric
```

## Core Circuit Components

**Name Mover Heads (NMH):**
- Layer 9-10 heads
- Copy IO token to END position
- Main heads: 9.9, 10.0, 9.6

**S-Inhibition Heads (SIH):** 
- Layer 7-8 heads
- Move duplicate token info from S2 to END
- Main heads: 7.3, 7.9, 8.6, 8.10

**Duplicate Token Heads (DTH):**
- Early layers (0-6)
- Detect duplicate tokens
- Use both positional & token information

## Validation Techniques

1. **Direct Attribution:** Project head outputs onto logit diff direction

2. **Attention Pattern Analysis:**
```python
cv.attention.attention_patterns(
    attention=patterns,
    tokens=tokens,
    head_names=names
)
```

3. **Activation/Path Patching:** Measure causal effects between components

4. **Ablation Studies:** Test performance when removing components

Remember: Always consider faithfulness, completeness, and minimality when validating circuit understanding.