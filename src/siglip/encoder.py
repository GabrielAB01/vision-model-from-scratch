import torch
import torch.nn as nn
from siglip.config import SiglipVisionConfig
from siglip.attention import SiglipAttention

from utils.load_weights import _copy_weights


class VisionEncoder(nn.Module):
	"""
		Implementation of the Vision Encoder Stack.

		Args:
			- config (SiglipVisionConfig): Configuration object for the model.
	"""
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		self.layers = nn.ModuleList([
			VisionEncoderLayer(config)
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
	
	def load_hf_weight(self, hf_state: dict, pbar=None):
		"""
			Chargement récursif des poids Hugging Face dans l'encodeur SigLIP.
			Args:
				hf_state : state-dict Hugging Face.
				pbar     : barre de progression (optionnelle).
		"""
		for idx, layer in enumerate(self.layers):
			layer.load_hf_weight(hf_state, idx, pbar=pbar)
		print(f"[INFO] Poids importés pour {self.__class__.__name__}")


class VisionEncoderLayer(nn.Module):
	"""
		Implementation of a single Siglip Encoder Layer.

		Args:
			- config (SiglipVisionConfig): Configuration object for the model.
	"""
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.embed_dim = config.hidden_size
		self.self_attn = SiglipAttention(config)
		# self.mlp = MLP(config)
		# Ne plus utiliser MLP, définir directement la séquence
		self.feed_forward = nn.Sequential(
			nn.Linear(config.hidden_size, config.intermediate_size),
			nn.GELU(approximate="tanh"),
			nn.Linear(config.intermediate_size, config.hidden_size),
		)

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
		# hidden_states = self.mlp(self.layer_norm2(hidden_states))
		hidden_states = self.feed_forward(self.layer_norm2(hidden_states))
		
		# Add residual connection
		hidden_states = residual + hidden_states
		return hidden_states 

	
	def load_hf_weight(self, hf_state: dict, layer_idx: int, pbar=None):
		"""
			Chargement des poids Hugging Face dans une couche de l’encodeur SigLIP.
			Args:
				hf_state  : state-dict Hugging Face déjà présent en mémoire.
				layer_idx : index (0-based) de la couche dans l’encodeur SigLIP.
				pbar      : barre de progression (optionnelle).
			"""

		# 1) Attention
		self.self_attn.load_hf_weight(hf_state, layer_idx, pbar=pbar)
		# 2) Feed-forward : faire le mapping de l'architecture HF vers notre architecture
		_copy_weights(
			self.feed_forward,
			hf_state,
			{
				"fc1.weight": "0.weight",
				"fc1.bias": "0.bias",
				"fc2.weight": "2.weight",
			 	"fc2.bias": "2.bias"
			},
			prefix_src=f"vision_tower.vision_model.encoder.layers.{layer_idx}.mlp.",
			pbar=pbar
		)
		# 3) LayerNorm1
		prefix = f"vision_tower.vision_model.encoder.layers.{layer_idx}."
		_copy_weights(
			self.layer_norm1,
			hf_state,
			{"weight": "weight", "bias": "bias"},
			prefix_src=prefix + "layer_norm1.",
			pbar=pbar
		)

		# 4) LayerNorm2
		_copy_weights(
			self.layer_norm2,
			hf_state,
			{"weight": "weight", "bias": "bias"},
			prefix_src=prefix + "layer_norm2.",
			pbar=pbar
		)

#------------------------------------------------------------------
# Classe MLP : Plus utilisée dans le modèle SigLIP, remplacée par nn.Sequential dans VisionEncoderLayer
#------------------------------------------------------------------
class MLP(nn.Module):
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
	
	def load_hf_weight(self, hf_state: dict, layer_idx: int, pbar=None):
		"""
			Chargement des poids Hugging Face dans une couche MLP de l'encodeur SigLIP.
			Args:
				hf_state  : state-dict Hugging Face déjà présent en mémoire.
				layer_idx : index (0-based) de la couche SigLIP dans l'encodeur.
				pbar      : barre de progression (optionnelle).
		"""

		# Préfixe exact des clés HF de cette couche MLP
		prefix = f"vision_tower.vision_model.encoder.layers.{layer_idx}.mlp."

		# Mapping 1-pour-1, biais inclus
		rename_map = {
			"fc1.weight": "fc1.weight",
			"fc1.bias":   "fc1.bias",
			"fc2.weight": "fc2.weight",
			"fc2.bias":   "fc2.bias",
		}

		_copy_weights(self, hf_state, rename_map, prefix_src=prefix, pbar=pbar)