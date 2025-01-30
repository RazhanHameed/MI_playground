# Balanced Bracket Classifier Cheatsheet

## Model Architecture & Setup

### Model Details
- **Task**: Classify bracket sequences as balanced/unbalanced
- **Architecture**: Bidirectional transformer (like BERT)
- **Size**: 3 layers, 2 heads per layer, d_model=56
- **Vocab**: 5 tokens: `[start]`, `[pad]`, `[end]`, `(`, `)`
- **Input Format**: `[start] + brackets + [end] + [pad...]`
- **Output**: Binary classification at position 0 (balanced/unbalanced)

### Key Differences from GPT
- Uses bidirectional attention (not causal)
- Only uses position 0 output for classification
- Pads sequences to fixed length with [pad] tokens
- Masks attention to [pad] tokens

## Model Components & Circuits

### Main Circuit: Total Elevation Detection
1. **Head 0.0**: Aggregation
   - Attends uniformly to all tokens after current position
   - Writes vectors v_L and v_R (cosine similarity ≈ -1)
   - Effectively counts difference between ( and )

2. **MLPs (0 & 1)**: Nonlinear Processing
   - Convert linear counts into boolean features
   - Detect when proportion != 0.5
   - Different neurons activate for >0.5 and <0.5

3. **Head 2.0**: Final Classification
   - Copies processed information from position 1 to 0
   - Strong attention to position 1
   - Output used for final classification

### Secondary Circuit: Negative Elevation Detection
- **Head 2.1**: Detects if any suffix has more ) than (
- Uses similar mechanism to elevation circuit
- MLPs process information at each position
- Combines with elevation circuit for final classification

## Analysis Techniques Used

### Logit Attribution
1. Get direction in residual stream indicating "unbalanced"
2. Decompose residual stream into component vectors
3. Measure components' dot product with unbalanced direction
4. Compare balanced vs unbalanced inputs

### Attention Pattern Analysis
- Pattern visualization via activation patching
- Query/key vector analysis
- Uniform attention indicates aggregation

### Linear Approximation of LayerNorm
- Fit linear regression to LayerNorm inputs/outputs
- High R² indicates good approximation
- Enables attribution through LayerNorm

### MLP Analysis
- View MLPs as collections of neurons
- Analyze input/output directions
- Plot neuron activations vs features (e.g., open proportion)

## Key Findings

### Model's Algorithm
1. Head 0.0 computes running total of bracket difference
2. MLPs convert this into boolean features
3. Head 2.0 moves this to position 0 for classification
4. Head 2.1 handles negative elevation failures
5. Combines both signals for final prediction

### Performance
- Near-perfect on standard cases
- Potential adversarial examples exist:
  - Very deep nesting
  - Carefully placed negative elevations
  - Early closing brackets

### Circuit Interactions
- Composition across layers
- Information flow through residual stream
- Multiple failure detection mechanisms
- Parallel processing of different features

## Common Tools & Methods

### TransformerLens Functions
- `run_with_cache()`: Get activations
- `get_act_name()`: Get activation names
- Permanent hooks for padding mask
- Activation patching

### Analysis Tools
- Cosine similarity
- Linear regression
- Attention pattern visualization
- Component attribution plots
- Neuron activation plots

### Visualization Functions
- `circuitsvis`: Attention patterns
- Histogram plots
- Scatter plots
- Bar charts for attention probabilities