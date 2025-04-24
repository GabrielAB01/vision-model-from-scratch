import torch
import torch.nn as nn
from siglip.config import SiglipVisionConfig
from siglip.attention import SiglipAttention


class SiglipEncoder(nn.Module):
	"""
		Implementation of the Siglip Encoder Stack.

		Args:
			- config (SiglipVisionConfig): Configuration object for the model.
	"""
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		self.layers = nn.ModuleList([
			SiglipEncoderLayer(config)
			for _ in range(config.num_hidden_layers)
		])

	def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
		"""
			Perform a forward pass through the Siglip Encoder.

			Args:
				input_embeddings (torch.Tensor): Input hidden states of shape [Batch_size, Num_Patches, Embed_Dim].

			Returns:
				torch.Tensor: The output of the model, of shape [Batch_size, Num_Patches, Embed_Dim].
		"""
		hidden_states = input_embeddings

		# Pass through each encoder layer
		for layer in self.layers:
			hidden_states = layer(hidden_states)
		
		return hidden_states


class SiglipEncoderLayer(nn.Module):
	"""
		Implementation of a single Siglip Encoder Layer.

		Args:
			- config (SiglipVisionConfig): Configuration object for the model.
	"""
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.embed_dim = config.hidden_size
		self.self_attn = SiglipAttention(config)
		self.mlp = SiglipMLP(config)

		self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
		self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		"""
			Perform a forward pass through the Siglip Encoder Layer.

			Args:
				hidden_states (torch.Tensor): Input hidden states of shape [Batch_size, Num_Patches, Embed_Dim].

			Returns:
				torch.Tensor: The output of the model, of shape [Batch_size, Num_Patches, Embed_Dim].
		"""
		# Self-attention (and layer normalization) sublayer
		residual = hidden_states
		hidden_states, _ = self.self_attn(self.layer_norm1(hidden_states))

		# Add & Norm
		hidden_states = residual + hidden_states
		residual = hidden_states

		# Feed-forward (and layer normalization) sublayer
		hidden_states = self.mlp(self.layer_norm2(hidden_states))
		
		# Add residual connection
		hidden_states = residual + hidden_states
		return hidden_states 


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states) # [Batch_Size, Num_Patches, Intermediate_Size]

		# Activation function : GELU
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
		
        hidden_states = self.fc2(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]

        return hidden_states