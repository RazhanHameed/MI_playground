# VAEs and GANs Cheat Sheet

## Autoencoders

### Basic Architecture
- **Encoder**: Compresses input into low-dimensional latent space
- **Decoder**: Reconstructs input from latent representation
- Loss function: Reconstruction loss (e.g., MSE) between input and output
- Common issue: Latent space may not be meaningful/continuous

### Typical Architecture Components
```python
# Encoder blocks:
Conv2d -> BatchNorm -> ReLU
Linear -> ReLU -> Linear (for latent space)

# Decoder blocks:
Linear -> ReLU -> Linear
ConvTranspose2d -> BatchNorm -> ReLU
```

## Variational Autoencoders (VAEs)

### Key Concepts
- Maps inputs to distributions (μ, σ) rather than points in latent space
- Uses reparameterization trick: z = μ + σ * ε where ε ~ N(0,1)
- Loss = Reconstruction Loss + KL Divergence Loss
- KL Loss = 0.5 * (σ² + μ² - 1 - log(σ²))

### Architecture Additions vs Autoencoder
```python
# Encoder outputs two vectors:
mu = self.fc_mu(x)
logsigma = self.fc_sigma(x)

# Reparameterization trick:
epsilon = torch.randn_like(mu)
z = mu + torch.exp(logsigma) * epsilon
```

### Training Tips
- Use β-VAE (β < 1) to balance reconstruction vs KL loss
- Monitor both reconstruction and KL divergence terms
- Check mean/std of latent distributions during training
- Visualize latent space interpolations

## GANs (Generative Adversarial Networks)

### Core Components
- **Generator**: Creates fake data from random noise
- **Discriminator**: Distinguishes real from fake data
- Adversarial training: Generator tries to fool discriminator

### Loss Functions
```python
# Discriminator loss (maximize log(D(x)) + log(1-D(G(z))))
D_x = discriminator(real_images)
D_G_z = discriminator(generated_images.detach())
loss_D = -torch.log(D_x).mean() - torch.log(1 - D_G_z).mean()

# Generator loss (minimize log(1-D(G(z))) or maximize log(D(G(z))))
D_G_z = discriminator(generated_images)
loss_G = -torch.log(D_G_z).mean()
```

### DCGAN Architecture Guidelines
1. Use strided convolutions instead of pooling
2. Use BatchNorm in both networks
3. Remove fully connected layers
4. Use ReLU in generator, LeakyReLU in discriminator
5. Use transposed convolutions for upsampling

### Training Tips
1. Initialize weights from N(0, 0.02)
2. Use separate optimizers for G and D
3. Use Adam with β1 = 0.5
4. Consider gradient clipping
5. Keep discriminator "winning" but not by too much
6. Monitor both D and G losses

## Transposed Convolutions

### Key Points
- Upsamples input by learned kernel
- Not literally inverse of convolution
- Output size = (input_size - 1) * stride + kernel_size - 2 * padding

### Implementation Tips
```python
# Core operation for 1D:
x_padded = pad1d(x, left=kernel_width-1, right=kernel_width-1)
weights_mod = weights.flip(-1)
return conv1d_minimal(x_padded, weights_mod)

# For stride > 1:
1. Space out input with zeros
2. Then perform regular transposed conv
```

## Common Issues & Solutions

### VAE Issues
- Reconstruction too blurry: Reduce β in KL loss
- Poor latent space: Increase β, check architecture
- Posterior collapse: Use cyclical annealing
- Mode collapse: Use more latent dimensions

### GAN Issues
- Mode collapse: Use batch diversity metrics
- Vanishing gradients: Use Wasserstein loss
- Training instability: 
  - Adjust learning rates
  - Use gradient clipping
  - Check batch norm momentum
  - Ensure proper weight initialization

## Debugging & Monitoring

### Weights & Biases Logging
```python
# Log images
images = [wandb.Image(img) for img in generated_images]
wandb.log({"samples": images}, step=step)

# Log metrics
wandb.log({
    "loss_D": loss_D,
    "loss_G": loss_G,
    "D_x": D_x.mean(),
    "D_G_z": D_G_z.mean()
}, step=step)
```

### Key Metrics to Monitor
- VAE: Reconstruction loss, KL divergence
- GAN: D loss, G loss, D(x), D(G(z))
- Image quality metrics (FID score if available)
- Gradient magnitudes
- Layer activation statistics