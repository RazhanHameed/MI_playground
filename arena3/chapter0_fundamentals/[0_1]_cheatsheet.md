# Ray Tracing Cheat Sheet

## Core Concepts

### Ray Representation
- A ray is represented by an origin point O and direction point D
- Ray equation: R(u) = O + uD where u ≥ 0
- Rays are stored in tensors of shape (num_rays, 2, 3) where:
  - First dimension indexes the rays
  - Second dimension contains [origin, direction] points
  - Third dimension contains [x, y, z] coordinates

### Line Segments
- Defined by two endpoints L₁ and L₂
- Segment equation: S(v) = L₁ + v(L₂ - L₁) where v ∈ [0,1]
- Intersection found by solving: O + uD = L₁ + v(L₂ - L₁)

### Triangles
- Defined by three vertices A, B, C
- Point inside triangle: P(u,v) = A + u(B-A) + v(C-A)
  where u ≥ 0, v ≥ 0, and u + v ≤ 1
- These u,v are called barycentric coordinates

## Key PyTorch Operations

### Tensor Creation & Manipulation
```python
# Create tensors
t.zeros((n, m))              # Create n×m tensor of zeros
t.ones((n, m))               # Create n×m tensor of ones
t.linspace(start, end, n)    # Create n evenly spaced points
t.stack([x, y, z], dim=0)    # Stack tensors along dimension

# Indexing
tensor[..., 0]               # Ellipsis to avoid repeated :
tensor.unbind(dim=1)         # Unpack tensor along dimension

# Shape operations
tensor.reshape(new_shape)    # Change tensor shape
tensor.view(new_shape)       # Change tensor shape (shares memory)
```

### Linear Algebra
```python
# Solve linear equations Ax = b
t.linalg.solve(A, b)         # Solve system of equations
t.linalg.det(A)              # Calculate determinant

# Vector operations
t.cross(a, b, dim=1)        # Cross product along dimension
```

### Logic Operations
```python
# Tensor logical operations
x & y                       # Element-wise AND
x | y                       # Element-wise OR
~x                         # Element-wise NOT
tensor.any(dim=1)          # Reduce with OR along dimension
tensor.all(dim=1)          # Reduce with AND along dimension
```

## Batched Operations

### Broadcasting Rules
- Dimensions are matched from right to left
- Size 1 dimensions are expanded to match larger size
- Missing dimensions are added on left

### Using einops
```python
# Repeat tensor along new dimensions
einops.repeat(x, 'a b -> a b c', c=4)

# Common patterns
rays = einops.repeat(rays, 'nrays p d -> nrays nsegments p d', nsegments=NS)
segments = einops.repeat(segments, 'nsegments p d -> nrays nsegments p d', nrays=NR)
```

## Implementation Tips

### Ray-Triangle Intersection
1. Form linear system:
```
[-D | B-A | C-A][s] = O-A
                 [u]
                 [v]
```
2. Solve for s, u, v
3. Check bounds:
   - s ≥ 0 (ray direction)
   - u ≥ 0, v ≥ 0 (inside triangle)
   - u + v ≤ 1 (inside triangle)

### Handling Edge Cases
- Check for singular matrices (det ≈ 0)
- Replace singular matrices with identity
- Set intersection distance to infinity for invalid cases

### Memory Efficiency
- Use views instead of copies when possible
- Basic indexing returns views
- Advanced indexing returns copies
- Use in-place operations when possible

## Testing & Debugging

### Using pytest
```python
@pytest.mark.parametrize("arg1, arg2", [
    (val1, val2),
    (val3, val4)
])
def test_function(arg1, arg2):
    ...
```


### Common Debugging Steps
1. Print tensor shapes
2. Check for NaN/Inf values
3. Verify matrix determinants
4. Test with simple cases first
5. Visualize intermediate results