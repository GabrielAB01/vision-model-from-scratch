from einops import rearrange
import torch
import torch.nn as nn

from typing import Tuple, Optional

from utils.load_weights import _copy_weights


class SiglipAttention(nn.Module):
	"""
		Multi-head self-attention module for the Siglip Vision Transformer.
		Args:
			- config (SiglipVisionConfig): Configuration object for the model.
	"""

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		assert self.embed_dim % self.num_heads == 0, (
			f"The embedding dimension {self.embed_dim} must be divisible by the number of heads {self.num_heads}."
		)
		self.head_dim = self.embed_dim // self.num_heads
		self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
		self.dropout = config.attention_dropout

		self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
		self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
		self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
		self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

	def forward(
		self,
		hidden_states: torch.Tensor,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

		# hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
		batch_size, seq_len, _ = hidden_states.size()
		
		# Projections Q ,K and V
		query_states = self.q_proj(hidden_states)
		key_states = self.k_proj(hidden_states)
		value_states = self.v_proj(hidden_states)

		# Reshape the query, key and value states to [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
		# query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		# key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		# value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

		# Reshape with einops : 
		# [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
		query_states = rearrange(query_states, 'b s (h d) -> b h s d', h=self.num_heads)
		key_states = rearrange(key_states, 'b s (h d) -> b h s d', h=self.num_heads)
		value_states = rearrange(value_states, 'b s (h d) -> b h s d', h=self.num_heads)
	

		# Calculate the attention using the formula Q * K^T / sqrt(d_k). 
		# attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
		# attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
		attn_weights = torch.einsum('bhsd,bhpd->bhsp', query_states, key_states) * self.scale

		if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
			raise ValueError(
				f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
				f" {attn_weights.size()}"
			)

		# Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
		attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
		# Apply dropout only during training
		attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
		
		# Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
		# attn_output = torch.matmul(attn_weights, value_states)
		
		# [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
		# attn_output = attn_output.transpose(1, 2).contiguous()
		# [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
		# attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

		# Avec einsum : produit et reshape
		# attn_weights : [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
		# value_states : [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
		# --> attn_output : [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
		attn_output = torch.einsum('bhsp,bhpd->bhsd', attn_weights, value_states)
		# [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
		attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')

		if attn_output.size() != (batch_size, seq_len, self.embed_dim):
			raise ValueError(
				f"`attn_output` should be of size {(batch_size, seq_len, self.embed_dim)}, but is"
				f" {attn_output.size()}"
			)
		
		# [Batch_Size, Num_Patches, Embed_Dim]
		attn_output = self.out_proj(attn_output)

		return attn_output, attn_weights
	
	def load_hf_weight(self, hf_state: dict, layer_idx: int, pbar=None):
		"""
			Chargement des poids Hugging Face dans la couche d'attention.
			Args:
				- hf_state  : state-dict Hugging Face déjà chargé en mémoire.
				- layer_idx : index (0-based) de la couche d'attention dans l'encodeur.
				- pbar      : barre de progression optionnelle.
		"""

		# Préfixe exact des clés HF pour cette couche d'attention
		prefix = f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn."

		# Mapping 1-pour-1 (et biais inclus, même s’ils sont déjà identiques)
		rename_map = {
			"q_proj.weight": "q_proj.weight",
			"k_proj.weight": "k_proj.weight",
			"v_proj.weight": "v_proj.weight",
			"out_proj.weight": "out_proj.weight",
			"q_proj.bias": "q_proj.bias",
			"k_proj.bias": "k_proj.bias",
			"v_proj.bias": "v_proj.bias",
			"out_proj.bias": "out_proj.bias",
		}

		_copy_weights(self, hf_state, rename_map, prefix_src=prefix, pbar=pbar)


# Test attention
if __name__ == "__main__":
	from config import SiglipVisionConfig
	batch_size = 1
	num_patches = 3
	dim_model = 8
	n_heads = 4
	config = SiglipVisionConfig(
		hidden_size=dim_model,
		num_attention_heads=n_heads,
		num_hidden_layers=6,
		num_channels=3,
		image_size=224,
		patch_size=16,
	)
	# Test the attention function
	# # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
	hidden_states = torch.randn(batch_size, num_patches, dim_model)
	print("Hidden states shape:", hidden_states.shape)  # Should be [batch_size, num_patches, dim_model]
	attn = SiglipAttention(config)
	output, attn_weights = attn(hidden_states)
	print("Attention output shape:", output.shape)  # Should be [batch_size, seq_len_1, dim_model]
	print(output)
