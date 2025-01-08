# Core SAE Concepts
- **Sparse Autoencoders (SAEs)**: Neural networks trained to reconstruct activations while enforcing sparsity
- **Purpose**: Disentangle features from superposition into interpretable components 
- **Structure**: Encoder maps input to higher-dim sparse latent space, decoder reconstructs input
- **Base Model Terms**:
  - Features = Characteristics of underlying data distribution
  - Activations = Model's internal representations
- **SAE Terms**:
  - Latents = Directions in SAE activation space (avoid term "features" to prevent confusion)
  - SAE Dashboard = Tool showing latent's activation distribution, logits, examples, etc.

# Training SAEs

## Key Config Parameters
1. **Data Generation**:
  - Model name, hook point, dataset path
  - Context size, batch size, prepend_bos flag
2. **Architecture**: 
  - d_sae (width), activation function
  - Weight initialization settings
3. **Training Hyperparameters**:
  - Learning rate, L1 coefficient
  - Warmup/decay steps for LR and L1
4. **Resampling Parameters**:
  - Feature sampling window
  - Dead feature threshold
  - Dead feature window size

## Training Tips
- Use WandB to track key metrics (L0, CE loss, explained variance)
- Balance reconstruction vs sparsity with L1 coefficient
- Avoid dead features through:
  - Appropriate learning rate
  - L1 warmup
  - Resampling (if needed)
- Watch for dense features (>1% activation frequency)
- Prefer Gated SAE architecture over standard
- Dataset should match base model's training distribution

# Understanding & Analyzing SAEs

## Core Analysis Methods
1. **Basic Methods**:
  - Max activating examples
  - Neuronpedia dashboards
  - Logit lens (projection onto unembedding)
2. **Advanced Methods**:
  - Direct logit attribution
  - Ablation studies 
  - Attribution patching
  - Latent-to-latent gradients
3. **Attention SAE Methods**:
  - Direct latent attribution
  - Source token analysis

## Common Patterns & Phenomena
- **Feature Splitting**: Single feature splitting into multiple more specific features in wider SAEs
- **Feature Absorption**: Features being "absorbed" into other more specific features
- **Layer-Specific Patterns**:
  - Early layers: Basic syntax/grammar
  - Middle layers: Complex semantics 
  - Late layers: Direct output influence

# Evaluating SAEs

## Key Metrics
- **Reconstruction**:
  - MSE loss
  - CE loss recovered
  - Explained variance
- **Sparsity**:
  - L0 (number of non-zero activations)
  - L1 (sum of absolute values)
  - Activation density histograms
- **Interpretability**:
  - Autointerp scores
  - Ablation studies
  - Circuit analysis results

## Common Issues
- Dead latents (never activate)
- Dense latents (activate too frequently)
- Poor reconstruction despite good sparsity
- Feature absorption making circuits hard to find
- Training on wrong distribution
- Insufficient width/training time

# Advanced Topics

## Architectures
- Standard SAE
- Gated SAE
- TopK SAE
- JumpReLU SAE
- End-to-end SAE
- Transcoders (for circuit analysis)
- Meta-SAEs (for disentangling absorbed features)

## Recent Developments
- GemmaScope (DeepMind's large-scale SAE release)
- Patch scoping for autointerp
- Automated circuit discovery
- Cross-model SAE transfer
- Feature absorption theory
