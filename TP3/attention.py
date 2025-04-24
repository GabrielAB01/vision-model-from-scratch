import torch
import torch.nn as nn
import numpy as np
import einops


def attention(
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        mask: torch.BoolTensor=None,
        dropout: nn.Dropout=None,
    ) -> tuple:
    """Computes multihead scaled dot-product attention from the
    projected queries, keys and values.

    Args
    ----
        q: Batch of queries.
            Shape of [batch_size, seq_len_1, n_heads, dim_model].
        k: Batch of keys.
            Shape of [batch_size, seq_len_2, n_heads, dim_model].
        v: Batch of values.
            Shape of [batch_size, seq_len_2, n_heads, dim_model].
        mask: Prevent tokens to attend to some other tokens (for padding or autoregressive attention).
            Attention is prevented where the mask is `True`.
            Shape of [batch_size, n_heads, seq_len_1, seq_len_2],
            or broadcastable to that shape.
        dropout: Dropout layer to use.

    Output
    ------
        y: Multihead scaled dot-attention between the queries, keys and values.
            Shape of [batch_size, seq_len_1, n_heads, dim_model].
        attn: Computed attention between the keys and the queries.
            Shape of [batch_size, n_heads, seq_len_1, seq_len_2].
    """
    qk = torch.einsum("b q n d , b k n d -> b n q k", q, k) / np.sqrt(q.shape[-1]) # Normalisation par sqrt(d)
    if mask is not None:
        # Appliquer le masque
        qk = qk.masked_fill(mask, -1e9)

    attn = torch.softmax(qk, dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    y = torch.einsum("b n q k, b k n d->b q n d", attn, v)

    return y, attn


class MultiheadAttention(nn.Module):
    """Multihead attention module.
    Can be used as a self-attention and cross-attention layer.
    The queries, keys and values are projected into multiple heads
    before computing the attention between those tensors.

    Parameters
    ----------
        dim: Dimension of the input tokens.
        n_heads: Number of heads. `dim` must be divisible by `n_heads`.
        dropout: Dropout rate.
    """
    def __init__(
            self,
            dim: int,
            n_heads: int,
            dropout: float,
        ):
        super().__init__()

        assert dim % n_heads == 0

        self.d_model = dim
        dk = dim // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.ModuleList()
        self.W_k = nn.ModuleList()
        self.W_v = nn.ModuleList()

        for i in range(n_heads):
            self.W_q.append(nn.Linear(dim, dk))
            self.W_k.append(nn.Linear(dim, dk))
            self.W_v.append(nn.Linear(dim, dk))

        self.W_out = nn.Linear(dim, dim)


    def forward(
            self,
            q: torch.FloatTensor,
            k: torch.FloatTensor,
            v: torch.FloatTensor,
            key_padding_mask: torch.BoolTensor = None,
            attn_mask: torch.BoolTensor = None,
        ) -> torch.FloatTensor:
        """Computes the scaled multi-head attention form the input queries,
        keys and values.

        Project those queries, keys and values before feeding them
        to the `attention` function.

        The masks are boolean masks. Tokens are prevented to attends to
        positions where the mask is `True`.

        Args
        ----
            q: Batch of queries.
                Shape of [batch_size, seq_len_1, dim_model].
            k: Batch of keys.
                Shape of [batch_size, seq_len_2, dim_model].
            v: Batch of values.
                Shape of [batch_size, seq_len_2, dim_model].
            key_padding_mask: Prevent attending to padding tokens.
                Shape of [batch_size, seq_len_2].
            attn_mask: Prevent attending to subsequent tokens.
                Shape of [seq_len_1, seq_len_2].

        Output
        ------
            y: Computed multihead attention.
                Shape of [batch_size, seq_len_1, dim_model].
        """
        # Création des masques par défaut (tous False) en cas d'absence d'un des masques
        mask_k = einops.rearrange(
            key_padding_mask if key_padding_mask is not None
            else torch.zeros((k.shape[0], k.shape[1]), dtype=torch.bool, device=k.device),
            "b k -> b 1 1 k"
        )
        mask_a = einops.rearrange(
            attn_mask if attn_mask is not None
            else torch.zeros((q.shape[1], k.shape[1]), dtype=torch.bool, device=q.device),
            "s1 s2 -> 1 1 s1 s2"
        )

        # Expansion des masques aux dimensions attendues
        mask_k = mask_k.expand(-1, self.n_heads, -1, -1)
        mask_a = mask_a.expand(q.shape[0], self.n_heads, -1, -1)

        # Combinaison des masques avec un OR logique
        mask = mask_k | mask_a

        # Projeter q, k, et v avec les couches linéaires W_q, W_k et W_v
        q = torch.stack([W_q(q) for W_q in self.W_q], dim=2)
        k = torch.stack([W_k(k) for W_k in self.W_k], dim=2)
        v = torch.stack([W_v(v) for W_v in self.W_v], dim=2)

        y, attn = attention(q, k, v, mask, self.dropout)

        y = einops.rearrange(y, "b q n d -> b q (n d)")

        y = self.W_out(y)

        return y
    

# Test
if __name__ == "__main__":
	batch_size = 2
	seq_len_1 = 5
	seq_len_2 = 7
	dim_model = 16
	n_heads = 4
     

    # Test the attention function
	q = torch.randn(batch_size, seq_len_1, n_heads, dim_model)
	k = torch.randn(batch_size, seq_len_2, n_heads, dim_model)
	v = torch.randn(batch_size, seq_len_2, n_heads, dim_model)
	
	mask = torch.zeros(batch_size, n_heads, seq_len_1, seq_len_2, dtype=torch.bool)
	mask[:, :, 2:, 3:] = True  # Example mask to prevent attending to some tokens
	dropout = nn.Dropout(0.1)
	y, attn = attention(q, k, v, mask, dropout)
	print("Attention output shape:", y.shape)  # Should be [batch_size, seq_len_1, n_heads, dim_model]
     
	# Test the MultiheadAttention class
	q = torch.randn(batch_size, seq_len_1, dim_model)
	k = torch.randn(batch_size, seq_len_2, dim_model)
	v = torch.randn(batch_size, seq_len_2, dim_model)

	attn_layer = MultiheadAttention(dim_model, n_heads, dropout=0.1)
	output = attn_layer(q, k, v)

	print("Output shape:", output.shape)  # Should be [batch_size, seq_len_1, dim_model]
     
