import torch
from transformer import MiniTransformer, visualize_attention
import numpy as np

def demo_transformer():
    # Model parameters
    vocab_size = 1000
    d_model = 16
    n_heads = 4
    d_ff = 64
    n_layers = 2
    
    # Create model
    model = MiniTransformer(vocab_size, d_model, n_heads, d_ff, n_layers)
    
    # Sample input sequence
    tokens = ["[START]", "Hello", "world", "!", "[END]"]
    token_ids = torch.tensor([[0, 1, 2, 3, 4]])  # Simplified token IDs
    
    # Forward pass
    output_repr, attention_maps = model(token_ids)
    
    print("Model Architecture:")
    print(f"- Embedding dimension: {d_model}")
    print(f"- Number of heads: {n_heads}")
    print(f"- Feed-forward dimension: {d_ff}")
    print(f"- Number of layers: {n_layers}")
    print(f"\nInput shape: {token_ids.shape}")
    print(f"Output shape: {output_repr.shape}")
    
    # Visualize attention patterns
    print("\nVisualizing attention patterns...")
    for layer in range(n_layers):
        for head in range(n_heads):
            visualize_attention(attention_maps, tokens, head_idx=head, layer_idx=layer)

if __name__ == "__main__":
    demo_transformer()
