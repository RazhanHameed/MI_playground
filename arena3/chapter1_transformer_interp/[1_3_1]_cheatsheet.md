# Superposition & Sparse Autoencoders Cheat Sheet

## Key Concepts

### Superposition
- When a model represents more than n features in an n-dimensional activation space
- Features correspond to directions, but there are more interpretable directions than dimensions
- Happens because models need to represent many features with limited dimensions

### Privileged vs Non-Privileged Basis
- **Privileged basis**: Standard basis directions are meaningful due to computation structure (e.g. neuron activations due to ReLU)
- **Non-privileged basis**: Can be rotated arbitrarily without changing computation (e.g. residual stream)

### Feature Properties
- **Importance**: How useful a feature is for achieving lower loss
- **Sparsity**: How infrequently it appears in input data (1 - feature probability)
- Higher sparsity allows more features to be packed into same dimensions with less interference

### Types of Feature Relationships
- **Correlated features**: Always appear together
- **Anticorrelated features**: Never appear together
- **Uncorrelated features**: Independent appearance
- Anticorrelated features easier to represent in superposition (less interference)

## Toy Models

### Basic Model Setup
```python
h = W x  # Map to lower dim
x' = ReLU(W^T h + b)  # Map back up
```

### Neuron Model Setup (Privileged Basis)
```python
h = ReLU(W x)  # Add ReLU for privileged basis
x' = ReLU(W^T h + b)
```

### Common Patterns
- Low sparsity: Features represented orthogonally or not at all
- High sparsity: More features represented non-orthogonally
- Correlated features: Represented in local orthogonal bases
- Anticorrelated features: Often represented as antipodal pairs

## Sparse Autoencoders (SAEs)

### Basic Architecture
```python
z = ReLU(W_enc(h - b_dec) + b_enc)  # Encode to higher dim
h' = W_dec z + b_dec  # Decode back
```

### Loss Function Components
- Reconstruction loss: MSE between input & output
- Sparsity penalty: L1 norm of hidden activations 
- Total loss = reconstruction_loss + l1_coeff * sparsity_loss

### Training Techniques
- Neuron resampling: Reset "dead" neurons that never activate
- Weight normalization: Normalize decoder weights 
- Larger latent dim than input (overcomplete)
- Can use tied or untied weights

### Gated Architecture
```python
# Gating term
active = (W_gate(x - b_dec) + b_gate > 0)
# Magnitude term  
magnitude = ReLU(W_mag(x - b_dec) + b_mag)
# Final output
z = active * magnitude
```

## Common Pitfalls/Tips

- Ensure proper weight initialization
- Monitor fraction of active neurons
- Use neuron resampling to prevent dead features
- Be careful with weight normalization
- Watch for shrinkage in standard SAEs

## Visualization Tools
- 2D feature plots for low-dim models
- Hidden state activation plots
- Dimensionality/capacity measures
- Fraction of active neurons over time
- Correlation matrices between features
