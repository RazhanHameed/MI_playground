

# Key NNsight Library Features 🔧

### Basic Setup & Model Loading
```python
from nnsight import LanguageModel
model = LanguageModel("EleutherAI/gpt-j-6b")
tokenizer = model.tokenizer
```

### Running Forward Passes
```python
# Single forward pass
with model.trace(prompt, remote=REMOTE):
    logits = model.lm_head.output[0, -1].save()  # Save final logits
    hidden = model.transformer.h[layer].output[0].save()  # Save hidden states

# Multiple forward passes
with model.trace(remote=REMOTE) as tracer:
    with tracer.invoke(prompts1):
        # First forward pass
    with tracer.invoke(prompts2): 
        # Second forward pass
```

### Multi-Token Generation
```python
with model.generate(max_new_tokens=n, remote=REMOTE) as generator:
    with generator.invoke(prompt):
        for _ in range(n):
            # Intervene before each token generation
            model.next()
```

# Function Vectors 🎯

### Extracting Function Vectors
1. Create ICL dataset with clean prompts showing desired task
2. Extract activations from key attention heads during clean forward pass
3. Average these activations across prompts to get function vector

### Using Function Vectors 
```python
# Basic intervention
with model.trace(prompt, remote=REMOTE):
    hidden = model.transformer.h[layer].output[0]
    hidden[:, -1] += function_vector # Add to final seq pos
```

### Key Applications
- Zero-shot task completion
- Steering multi-token generation
- Task composition through vector addition
- Analyzing decoded vocabulary 

# Steering Vectors 🚗

### Basic Structure
```python
activation_additions = [
    (layer1, coef1, "prompt1"),
    (layer2, coef2, "prompt2")
]
```

### Common Use Cases
- Changing emotional tone (positive/negative)
- Altering factual beliefs
- Modifying behavioral patterns

### Best Practices
- Use contrast pairs when possible
- Test different layers & coefficients
- Consider using sampling params like top_p
- Add repetition penalty for better coherence

# Important Tips 📝

### Memory Management
- Only save necessary tensors with .save()
- Clear memory between large batches
- Use smaller batch sizes for large models