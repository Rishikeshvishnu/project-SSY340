#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from math import sqrt

from data_preprocessing import *


# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# In[3]:


# function for absolute positional encoding

def sinusoidal_positional_encoding(max_len, dim_model, dim_op=3):
    """
    Creates sinusoidal position embeddings for transformer inputs. Each position is encoded
    using sine for even dimensions and cosine for odd dimensions at different frequencies.
     
    Args:
        max_len (int): Maximum sequence length to generate positions for
        dim_model (int): Size of each position encoding vector (must be even)
        dim_op (int): Target output dimension, default 3 for (batch, seq_len, dim_model)
    
    Returns:
        torch.Tensor: Position encodings of shape ([1,]*dim_op-2, max_len, dim_model)
                     Ready for broadcasting when added to input embeddings
    """
        
    # Create position and dimension indices
    positions = torch.arange(max_len, dtype=torch.float).to(device)
    dimensions = torch.arange(0, dim_model, 2, dtype=torch.float).to(device)
    
    # Calculate angle rates
    angle_rates = 1 / (10000 ** (dimensions / dim_model))
    
    # Calculate angles by outer product
    angles = positions.unsqueeze(1) * angle_rates.unsqueeze(0)
    
    # Apply sin/cos
    encodings = torch.zeros(max_len, dim_model)
    encodings[:, 0::2] = torch.sin(angles)
    encodings[:, 1::2] = torch.cos(angles)
    
    # Reshape for broadcasting
    shape = [1] * (dim_op-2) + [max_len, dim_model]
    return encodings.view(*shape)


# In[4]:


class MultiHeadRelativeAttention(nn.Module):
    """
    Multi-Head Attention with Relative Position Embeddings.
    """
    def __init__(self, dim_model: int, num_attention_heads: int, max_relative_position: int, use_bias: bool = True):
        """
        Args:
            dim_model: Model's hidden dimension size (must be divisible by num_attention_heads)
            num_attention_heads: Number of parallel attention heads
            max_relative_position: Maximum relative distance for position embeddings
            use_bias: Whether to include bias terms in linear transformations
        """
        super().__init__()
        
        # Validate dimensions
        if dim_model % num_attention_heads != 0:
            raise ValueError("dim_model must be divisible into num_attention_heads")
        
        # Core dimensions
        self.dim_model = dim_model
        self.num_heads = num_attention_heads
        self.head_dim = dim_model // num_attention_heads
        self.max_relative_position = max_relative_position
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(dim_model, dim_model, bias=use_bias)
        self.key_proj = nn.Linear(dim_model, dim_model, bias=use_bias)
        self.value_proj = nn.Linear(dim_model, dim_model, bias=use_bias)
        
        # Relative position embeddings
        self.rel_emb = nn.Embedding(max_relative_position, dim_model)
        
        # Output projection
        self.output_proj = nn.Linear(dim_model, dim_model, bias=True)
    
    def reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes input tensor into multiple attention heads.
        Args:
            x: tensor of shape (..., L, dim_model)
        Returns:
            reshaped tensor of shape (..., num_heads, L, head_dim)
        """
        x = x.view(*x.shape[:-1], self.num_heads, self.head_dim)
        return x.transpose(-2, -3)
    
    def get_relative_positions(self, seq_length: int, max_len: Optional[int] = None) -> torch.Tensor:
        """
        Generates relative position embeddings for the sequence.
        
        Args:
            seq_length: Length of input sequence
            max_len: Maximum relative distance to consider (default: self.max_relative_position)
        """
        if max_len is None:
            max_len = self.rel_emb.num_embeddings
            
        device = self.rel_emb.weight.device
        
        # Generate base embeddings
        base_embedding = self.rel_emb(torch.arange(0, 1, device=device)).clone()
        padding_size = max(seq_length - max_len, 0)
        
        # Generate required embeddings sequence
        relative_positions = torch.cat([
            *[base_embedding.clone() for _ in range(padding_size)],
            self.rel_emb(
                torch.arange(
                    max(max_len - seq_length, 0),
                    max_len,
                    device=device
                )
            )
        ], dim=0)
        
        return relative_positions

    def compute_relative_attention(self, query: torch.Tensor, key: torch.Tensor, 
                                 value: torch.Tensor, rel_pos_emb: Optional[torch.Tensor] = None,
                                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes relative attention with optional position-based bias and masking.
        
        Args:
            query: Query tensor of shape (..., seq_len, d_k)
            key: Key tensor of shape (..., seq_len, d_k)
            value: Value tensor of shape (..., seq_len, d_v)
            rel_pos_emb: Relative positional embeddings of shape (..., seq_len, d_k)
            mask: Attention mask tensor, where 1 indicates positions to mask
            
        Returns:
            torch.Tensor: Output tensor after applying attention, shape (..., seq_len, d_v)
        """
        # Compute content-based attention scores
        content_scores = torch.matmul(query, key.transpose(-1, -2))

        # Compute position-based scores if relative positional embedding is provided
        if rel_pos_emb is not None:
            position_raw_scores = torch.matmul(query, rel_pos_emb.transpose(-1, -2))
            
            # Align relative positions by padding, reshaping, and removing first column
            padded_pos_scores = F.pad(position_raw_scores, [1, 0])
            transformed = padded_pos_scores.reshape(-1, position_raw_scores.shape[-1] + 1, position_raw_scores.shape[-2])
            position_scores = transformed[:, 1:].reshape(*position_raw_scores.shape)
        else:
            position_scores = torch.zeros_like(content_scores)

        # Scale combined scores
        scale = 1 / sqrt(query.size(-1))
        attention_scores = (content_scores + position_scores) * scale

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 1, -1e9)

        # Compute attention weights and apply to values
        attention_weights = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, value)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-head relative attention.
        
        Args:
            query: Query tensor (batch_size, seq_len_q, dim_model)
            key: Key tensor (batch_size, seq_len_k, dim_model)
            value: Value tensor (batch_size, seq_len_k, dim_model)
            attention_mask: Optional mask tensor with 1s indicating positions to mask
            
        Returns:
            Attention output tensor (batch_size, seq_len_q, dim_model)
        """
        # Project inputs
        query = self.query_proj(query)  # (batch_size, seq_len_q, dim_model)
        key = self.key_proj(key)        # (batch_size, seq_len_k, dim_model)
        value = self.value_proj(value)  # (batch_size, seq_len_k, dim_model)
        
        # Get relative position embeddings
        rel_pos = self.get_relative_positions(key.shape[-2], self.max_relative_position)  # (seq_len_k, dim_model)
        
        # Reshape inputs for multi-head attention
        query_heads = self.reshape_for_heads(query)        # (batch_size, num_heads, seq_len_q, head_dim)
        key_heads = self.reshape_for_heads(key)           # (batch_size, num_heads, seq_len_k, head_dim)
        value_heads = self.reshape_for_heads(value)       # (batch_size, num_heads, seq_len_k, head_dim)
        rel_pos_heads = self.reshape_for_heads(rel_pos)   # (num_heads, seq_len_k, head_dim)
        
        # Compute attention for all heads together using the class method
        attention_output = self.compute_relative_attention(
            query_heads,
            key_heads,
            value_heads,
            rel_pos_heads,
            mask=attention_mask
        )
        
        # Reshape output back to original dimensions 
        attention_output = attention_output.transpose(-2, -3)  # (batch_size, seq_len_q, num_heads, head_dim)
        concatenated = attention_output.reshape(*attention_output.shape[:-2], self.dim_model)
           
        return self.output_proj(concatenated)


# In[5]:


class FeedForward(nn.Module):
    """
    Feed Forward Network with two linear transformations and GeLU activation.
    """
    def __init__(self, dim_model: int, dim_ff: int, dropout: float = 0.1, bias: bool = True):
        """
        Args:
            dim_model: Input and output dimension
            dim_ff: Hidden dimension of the feed-forward network
            dropout: Dropout rate
            bias: Whether to include bias terms in linear transformations
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_model, dim_ff, bias=bias),
            nn.GELU(),  
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_model, bias=bias)
        )
        
        # Initialize weights using Kaiming initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:  
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim_model)
        Returns:
            Output tensor of shape (batch, seq_len, dim_model)
        """
        return self.network(x)


# In[6]:


def create_mask(inp, n=4):
    """
    Creates a combined mask that handles both padding and look-ahead masking for transformer inputs.
    The mask prevents the transformer from attending to padding tokens and future tokens in the sequence.
    Positions to be masked are marked with ones in the output tensor.

    Args:
        inp: unembedded batch of input sequences of shape (batch_size, seq_len)
        n (int): number of dimensions to which to broadcast mask

    Returns:
        combined_mask: maximum of padding and look_ahead masks for inp;
                      tensor of ones of shape (batch_size, 1, ..., 1, seq_len, seq_len) with ndim=n
                      positions to mask are marked with ones
    """
    # Create padding mask
    # Find positions in inp equal to pad_token
    padding_mask = torch.eq(inp, pad_token).float()
    # Add extra dimensions to padding mask
    padding_mask = padding_mask.view(
        *padding_mask.shape[:-1], 
        *[1 for _ in range(n-2)], 
        padding_mask.shape[-1]
    ).to(inp.device)

    # Create look ahead mask
    # Create upper triangular mask of ones
    seq_len = inp.shape[-1]
    look_ahead_mask = torch.triu(
        torch.ones(seq_len, seq_len), 
        diagonal=1
    ).float().to(inp.device)

    # Combine masks by taking maximum
    # Final mask is the maximum of padding and look ahead masks
    combined_mask = torch.max(padding_mask, look_ahead_mask)
    
    return combined_mask


# In[7]:


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block with relative positional self-attention using Pre-LN architecture.
    Architecture: Layer norm -> Multi head rel-attention -> Add -> Layer norm -> FFN -> Add
    """
    def __init__(
        self, 
        dim_model: int,
        num_heads: int,
        dim_ff: int,
        max_rel_dist: int,
        bias: bool = True,
        dropout: float = 0.1,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.max_rel_dist = max_rel_dist
        
        # Main layers
        self.self_attention = MultiHeadRelativeAttention(
            dim_model=dim_model,  
            num_attention_heads=num_heads,
            max_relative_position=max_rel_dist,
            use_bias=bias  
        )
        
        self.feed_forward = FeedForward(
            dim_model=dim_model,
            dim_ff=dim_ff,
            dropout=dropout,
            bias=bias
        )
        
        # Normalization and regularization
        self.norm1 = nn.LayerNorm(dim_model, eps=eps)
        self.norm2 = nn.LayerNorm(dim_model, eps=eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self, 
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Args:
            tgt: Input tensor of shape (batch, seq_len, dim_model)
            memory: Unused, kept for compatibility
            tgt_mask: Optional attention mask tensor
            Other args kept for compatibility with torch.nn.TransformerDecoder
            
        Returns:
            Output tensor of shape (batch, seq_len, dim_model)
        """
        # Self-attention block with Pre-LN and residual connection
        normed = self.norm1(tgt)
        attended = self.self_attention(
            query=normed, 
            key=normed, 
            value=normed, 
            attention_mask=tgt_mask
        )
        attn_out = tgt + self.dropout1(attended)
        
        # Feed-forward block with Pre-LN and residual connection
        normed = self.norm2(attn_out)
        transformed = self.feed_forward(normed)
        output = attn_out + self.dropout2(transformed)
        
        return output


# In[ ]:




