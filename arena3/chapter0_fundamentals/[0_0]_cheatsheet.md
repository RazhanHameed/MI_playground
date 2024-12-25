# Tensor Manipulation Cheatsheet

## Einops Operations

### rearrange
Rearranges dimensions without changing the data

```python
# Basic dimension reordering
x = einops.rearrange(tensor, "batch channel height width -> channel height (batch width)")

# Splitting dimensions
x = einops.rearrange(tensor, "(batch1 batch2) channels h w -> (batch1 h) (batch2 w) channels", batch1=2)

# Combining dimensions
x = einops.rearrange(tensor, "b c (h h2) (w w2) -> b c h w (h2 w2)", h2=2, w2=2)
```

### repeat
Repeats data along specified dimensions
```python
# Repeat along single dimension
x = einops.repeat(tensor, "c h w -> c (2 h) w")

# Repeat along multiple dimensions
x = einops.repeat(tensor, "c h w -> c (2 h) (3 w)")

# Add new dimensions
x = einops.repeat(tensor, "h w -> 1 h w 1")
```

### reduce
Reduces dimensions using an operation
```python
# Common reduction operations
mean = einops.reduce(tensor, "b c h w -> h w", "mean")
max_val = einops.reduce(tensor, "b c h w -> h w", "max")
sum_val = einops.reduce(tensor, "b c h w -> c", "sum")

# Reducing multiple dimensions
x = einops.reduce(tensor, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2)
```

## Einsum Operations
Uses Einstein summation convention for tensor operations

### Basic Operations
```python
# Matrix multiplication
C = einops.einsum(A, B, "i j, j k -> i k")

# Batch matrix multiplication
C = einops.einsum(A, B, "b i j, b j k -> b i k")

# Inner product
c = einops.einsum(a, b, "i, i ->")

# Outer product
C = einops.einsum(a, b, "i, j -> i j")

# Element-wise multiplication
C = einops.einsum(A, B, "i j, i j -> i j")

# Matrix trace
t = einops.einsum(A, "i i ->")
```

## Tensor Indexing & Manipulation

### Advanced Indexing
```python
# Integer array indexing
values = matrix[coords.T]  # coords shape: (batch, dims)

# Gathering values
output = matrix.gather(1, indices)  # gather along dim 1

# Batched gathering
output = matrix.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
```

### Common Operations
```python
# Logsumexp (numerically stable)
def logsumexp(x, dim):
    max_x = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return max_x + torch.log(exp_x.sum(dim=dim, keepdim=True))

# Softmax (numerically stable)
def softmax(x, dim):
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

# Cross entropy loss
def cross_entropy(logits, labels):
    log_probs = log_softmax(logits, dim=-1)
    return -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
```
