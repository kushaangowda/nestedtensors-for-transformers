import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout_p (float, optional): Dropout probability. Default: 0.0
    """
    def __init__(self, E_q: int, E_k: int, E_v: int, E_total: int,
                 nheads: int, dropout_p: float = 0.0):
        super().__init__()
        self.nheads = nheads
        self.dropout_p = dropout_p
        self.query_proj = nn.Linear(E_q, E_total)
        self.key_proj = nn.Linear(E_k, E_total)
        self.value_proj = nn.Linear(E_v, E_total)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_t, E_q)
            key (torch.Tensor): key of shape (N, L_s, E_k)
            value (torch.Tensor): value of shape (N, L_s, E_v)

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        # TODO: demonstrate packed projection
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout_p, is_causal=True)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output

class TransformerEncoderWithNested(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerEncoderWithNested, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, embed_dim, embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, nested_input):
        attn_output = self.attention(nested_input, nested_input, nested_input)
        nested_input = self.norm1(nested_input + attn_output)
        ff_output = self.ff(nested_input)
        nested_input = self.norm2(nested_input + ff_output)
        return nested_input
    
class TransformerEncoderWithPadding(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerEncoderWithPadding, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, embed_dim, embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.transpose(0, 1)
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x.transpose(0, 1)