import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Convert token IDs to embeddings."""
        return self.embedding(token_ids)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:, :x.size(1)]

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        
        return output, attn_weights

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation=nn.ReLU):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation()
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(d_model, n_heads)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention sublayer
        attn_out, attn_weights = self.self_attn(x, mask)
        x = x + attn_out  # Residual connection
        x = self.attn_norm(x)
        
        # Feed-forward sublayer
        ffn_out = self.ffn(x)
        x = x + ffn_out  # Residual connection
        x = self.ffn_norm(x)
        
        return x, attn_weights

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        
    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor = None):
        x = self.embed(token_ids)
        x = self.pos_enc(x)
        attention_maps = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_maps.append(attn_weights)
            
        return x, attention_maps

def visualize_attention(attention_weights, tokens, head_idx=0, layer_idx=0):
    """Visualize attention weights for a specific head in a specific layer."""
    attn = attention_weights[layer_idx][0, head_idx].detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    
    # Add token labels
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    
    plt.title(f'Attention Weights (Layer {layer_idx+1}, Head {head_idx+1})')
    plt.tight_layout()
    plt.show()
