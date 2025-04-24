import torch
import torch.nn as nn
from attention import MultiheadAttention


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer.

    Parameters
    ----------
        d_model: The dimension of input tokens.
        dim_feedforward: Hidden dimension of the feedforward networks.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            nhead: int,
            dropout: float,
        ):
        super().__init__()

        # Self-attention sub-layer
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)

        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.FloatTensor,
        key_padding_mask: torch.BoolTensor
        ) -> torch.FloatTensor:
        """Encodes the input. Does not attend to masked inputs.

        Args
        ----
            src: Batch of embedded source tokens.
                Shape of [batch_size, src_seq_len, dim_model].
            key_padding_mask: Mask preventing attention to padding tokens.
                Shape of [batch_size, src_seq_len].

        Output
        ------
            y: Batch of encoded source tokens.
                Shape of [batch_size, src_seq_len, dim_model].
        """
        # Self-attention sublayer
        attn_out = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)
        src = self.norm1(src + self.dropout(attn_out))
        feed_forward_out = self.feed_forward(src)
        y = self.norm2(src + self.dropout(feed_forward_out))
        return y


class TransformerEncoder(nn.Module):
    """Implementation of the transformer encoder stack.

    Parameters
    ----------
        d_model: The dimension of encoders inputs.
        dim_feedforward: Hidden dimension of the feedforward networks.
        num_encoder_layers: Number of stacked encoders.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            dim_feedforward: int,
            num_encoder_layers: int,
            nheads: int,
            dropout: float
        ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, dim_feedforward, nheads, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)


    def forward(
            self,
            src: torch.FloatTensor,
            key_padding_mask: torch.BoolTensor
        ) -> torch.FloatTensor:
        """Encodes the source sequence by sequentially passing.
        the source sequence through the encoder stack.

        Args
        ----
            src: Batch of embedded source sentences.
                Shape of [batch_size, src_seq_len, dim_model].
            key_padding_mask: Mask preventing attention to padding tokens.
                Shape of [batch_size, src_seq_len].

        Output
        ------
            y: Batch of encoded source sequence.
                Shape of [batch_size, src_seq_len, dim_model].
        """
        for layer in self.layers:
            src = layer(src, key_padding_mask)

        y = self.norm(src)
        return y
    


# Encoder Test
if __name__ == "__main__":
	# Test the TransformerEncoder
	encoder = TransformerEncoder(d_model=512, dim_feedforward=2048, num_encoder_layers=6, nheads=8, dropout=0.1)
	src = torch.randn(32, 10, 512)  # Batch of 32 sequences of length 10 with dimension 512
	key_padding_mask = torch.zeros(32, 10).bool()  # No padding in this example

	output = encoder(src, key_padding_mask)
	print(output.shape)  # Should be [32, 10, 512]