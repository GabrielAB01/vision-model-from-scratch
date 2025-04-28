import torch
from torch import nn
from typing import Optional

from gemma.config import GemmaConfig
from gemma.attention import KVCache, GemmaAttention
from utils.load_weights import _copy_weights

class GemmaDecoderLayer(nn.Module):
	"""
		Implémentation d'une couche du décodeur du modèle Gemma.
		Args:
			config (GemmaConfig): Configuration du modèle.
			layer_idx (int): Index de la couche.
	"""
	def __init__(self, config: GemmaConfig, layer_idx: int):
		super().__init__()
		
		self.hidden_size = config.hidden_size

		self.self_attn = GemmaAttention(config, layer_idx)

		self.mlp = GemmaMLP(config)
		self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	
	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		kv_cache: Optional[KVCache] = None,
	) -> torch.FloatTensor:
		"""
		Forward pass through the decoder layer.
		Args:
			hidden_states (torch.Tensor): The input hidden states. [Batch_Size, Seq_Len, Hidden_Size]
			attention_mask (torch.Tensor): The attention mask. [Batch_Size, Seq_Len]
			position_ids (torch.LongTensor): The position IDs. [Batch_Size, Seq_Len]
			kv_cache (Optional[KVCache]): The key-value cache for autoregressive decoding.
		Returns:
			torch.FloatTensor: The output hidden states after the decoder layer. [Batch_Size, Seq_Len, Hidden_Size]
		"""
		
		# --------- Self Attention --------- 
		residual = hidden_states # [Batch_Size, Seq_Len, Hidden_Size]

		hidden_states = self.input_layernorm(hidden_states)

		hidden_states, _ = self.self_attn(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			position_ids=position_ids,
			kv_cache=kv_cache,
		)
		hidden_states = hidden_states + residual

		# --------- MLP ---------
		residual = hidden_states # [Batch_Size, Seq_Len, Hidden_Size]
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states = self.mlp(hidden_states)
		hidden_states = hidden_states + residual

		return hidden_states
	
	def load_hf_weight(self, hf_state, layer_idx: int, pbar=None):
		"""
		Charge récursivement les poids du modèle Hugging Face dans la couche.
		Args:
			hf_state (dict): État du modèle Hugging Face.
			layer_idx (int): Index de la couche.
			pbar (Optional): Barre de progression.
		"""
		# 1) Attention
		self.self_attn.load_hf_weight(hf_state, layer_idx, pbar=pbar)

		# 2) MLP
		self.mlp.load_hf_weight(hf_state, layer_idx, pbar=pbar)

		# 3) RMSNorms
		prefix = f"language_model.model.layers.{layer_idx}."
		self.input_layernorm.load_hf_weight(hf_state, prefix + "input_layernorm.", pbar=pbar)
		self.post_attention_layernorm.load_hf_weight(hf_state, prefix + "post_attention_layernorm.", pbar=pbar)




class GemmaRMSNorm(nn.Module):
	"""
		Implémentation de la normalisation RMS.
		Args:
			hidden_size (int): Taille de l'espace caché.
			eps (float): Valeur epsilon pour la normalisation.
	"""
	def __init__(self, hidden_size: int, eps: float = 1e-6):
		super().__init__()

		self.eps = eps
		self.weight = nn.Parameter(torch.zeros(hidden_size))

	def _norm(self, x: torch.Tensor) -> torch.Tensor:
		# Normalisation RMS
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
	

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		output = self._norm(x.float())
		output = output * (1.0 + self.weight.float())
		return output.type_as(x)
	
	# Old function
	def _load_hf_weight(self, hf_state, layer_idx: int, which: str):
		"""
		layer_idx : index de la couche (0-based)
		which     : "input_layernorm." ou "post_attention_layernorm."
		"""
		prefix = f"language_model.model.layers.{layer_idx}.{which}"
		rename_map = {"weight": "weight"}
		_copy_weights(self, hf_state, rename_map, prefix_src=prefix)

	def load_hf_weight(self, hf_state, prefix: str, pbar=None):
		"""
		layer_idx : index de la couche (0-based)
		prefix    : préfixe huggingface complet
		pbar      : Barre de progression.
		"""
		rename_map = {"weight": "weight"}
		_copy_weights(self, hf_state, rename_map, prefix_src=prefix, pbar=pbar)
	
	
class GemmaMLP(nn.Module):
	"""
		Implémentation de la couche MLP du modèle Gemma.
		Args:
			config (GemmaConfig): Configuration du modèle.
	"""
	def __init__(self, config: GemmaConfig):
		super().__init__()

		self.hidden_size = config.hidden_size
		self.intermediate_size = config.intermediate_size

		self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
		self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
		self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through the MLP layer.
		Args:
			x (torch.Tensor): The input tensor. [Batch_Size, Seq_Len, Hidden_Size]
		Returns:
			torch.Tensor: The output tensor after the MLP layer. [Batch_Size, Seq_Len, Hidden_Size]
		"""
		
		# [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
		y = self.gate_proj(x) 
		y = nn.functional.gelu(y, approximate="tanh")

		# [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
		j = self.up_proj(x)
		z = y * j

		# [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
		z = self.down_proj(z)

		# return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
		return z
	
	def load_hf_weight(self, hf_state, layer_idx: int, pbar=None):
		prefix = f"language_model.model.layers.{layer_idx}.mlp."
		rename_map = {
			"gate_proj.weight": "gate_proj.weight",
			"up_proj.weight"  : "up_proj.weight",
			"down_proj.weight": "down_proj.weight",
		}
		_copy_weights(self, hf_state, rename_map, prefix_src=prefix, pbar=pbar)
