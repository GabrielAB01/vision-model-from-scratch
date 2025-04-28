import torch
import torch.nn as nn
import einops

from siglip.config import SiglipVisionConfig
from siglip.encoder import VisionEncoder

from utils.load_weights import _copy_weights

class SiglipVisionModel(nn.Module):
	"""
		Implementation of the Siglip Vision Model.

		Args:
			- config (dict): Configuration dictionary containing model parameters.
	"""
	def __init__(self, config: SiglipVisionConfig):
		"""
			Initialize the Siglip Vision Model.

			Args:
				config (SiglipVisionConfig): Configuration object for the model.
		"""
		# Call the parent constructor
		# and initialize the model with the given configuration
		super().__init__()
		self.config = config
		self.vision_model = VisionTransformer(config)  # Initialize the vision model with the config

	def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
		"""
			Perform a forward pass through the model.

			Args:
				pixel_values (torch.Tensor): Input pixel values of shape [Batch_size, Channels, Height, Width].

			Returns:
				torch.Tensor: The output of the model.
		"""
		# [Batch_size, Channels, Height, Width] -> [Batch_size, Num_Patches, Embed_Dim]
		return self.vision_model(pixel_values)
	
	def load_hf_weight(self, hf_state: dict, pbar=None):
		# Tout le travail est délégué au transformer interne
		self.vision_model.load_hf_weight(hf_state, pbar=pbar)
		print(f"[INFO] Poids importés pour {self.__class__.__name__}")
	

class VisionTransformer(nn.Module):
	"""
		Implementation of the Siglip Vision Transformer.

		Args:
			- config (SiglipVisionConfig): Configuration object for the model.
	"""
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config

		self.embeddings = SiglipVisionEmbeddings(config)
		self.encoder = VisionEncoder(config)
		self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
	
	
	def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
		"""
			Perform a forward pass through the Siglip Vision Transformer.

			Args:
				pixel_values (torch.Tensor): Input pixel values of shape [Batch_size, Channels, Height, Width].

			Returns:
				torch.Tensor: The output of the model.
		"""
		# [Batch_size, Channels, Height, Width] -> [Batch_size, Num_Patches, Embed_Dim]
		embeddings = self.embeddings(pixel_values)
		
		# Pass through the encoder
		outputs = self.encoder(embeddings)
		
		return self.post_layernorm(outputs)
	
	def load_hf_weight(self, hf_state: dict, pbar=None):
		# Charge les poids de Hugging Face pour le modèle Siglip Vision Transformer.

		# 1) Embeddings (conv patch + positions)
		self.embeddings.load_hf_weight(hf_state, pbar=pbar)
		# 2) Encodeur (N couches)
		self.encoder.load_hf_weight(hf_state, pbar=pbar)

		# 3) LayerNorm final
		_copy_weights(
			self.post_layernorm,
			hf_state,
			{"weight": "weight", "bias": "bias"},
			prefix_src="vision_tower.vision_model.post_layernorm.",
			pbar=pbar,
		)
		print(f"[INFO] Poids importés pour {self.__class__.__name__}")
	

class SiglipVisionEmbeddings(nn.Module):
	"""
		Implementation of the Siglip Vision Embeddings.

		Args:
			- config (SiglipVisionConfig): Configuration object for the model.
	"""
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		self.embed_dim = config.hidden_size
		self.image_size = config.image_size
		self.patch_size = config.patch_size

		# Convolutional layer to create patch embeddings : The kernel size and stride are both equal to the patch size
		# This means that the convolution will not overlap, and each patch will be of size [patch_size, patch_size]
		self.patch_embedding = nn.Conv2d(
			in_channels=config.num_channels,
			out_channels=self.embed_dim,
			kernel_size=self.patch_size,
			stride=self.patch_size,
			padding="valid", # No padding
		)

		assert self.image_size % self.patch_size == 0, (
			f"Image size ({self.image_size}) must be divisible by patch size ({self.patch_size})."
		)
		# Calculate the number of patches
		self.num_patches = (self.image_size // self.patch_size) ** 2

		# Position encoding for the patches
		self.num_positions = self.num_patches
		self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
		self.register_buffer(
			"position_ids",
			torch.arange(self.num_positions).expand((1, -1)),
			persistent=False,
		)

	def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
		"""
			Perform a forward pass through the Siglip Vision Embeddings.

			Args:
				pixel_values (torch.FloatTensor): Input pixel values of shape [Batch_size, Channels, Height, Width].

			Returns:
				torch.Tensor: The output of shape [Batch_ize, Num_Patches, Embed_Dim].
		"""
		
		# The output of the convolution will have shape [Batch_size, Embed_Dim, Num_Patches_H, Num_Patches_W]
		# where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
		patch_embeds = self.patch_embedding(pixel_values)  

		# [Batch_size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_size, Num_Patches, Embed_Dim]
		embeddings = einops.rearrange(patch_embeds, "b e h w -> b (h w) e")

		"""
		# [Batch_size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_size, Embed_Dim, Num_Patches]
		# where Num_Patches = Num_Patches_H * Num_Patches_W
		embeddings = patch_embeds.flatten(2)
		# [Batch_size, Embed_Dim, Num_Patches] -> [Batch_size, Num_Patches, Embed_Dim]
		embeddings = embeddings.transpose(1, 2)
		"""
		# Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
		embeddings = embeddings + self.position_embedding(self.position_ids)
		
		return embeddings

	def load_hf_weight(self, hf_state: dict, pbar=None):
		"""
			Charge les poids de Hugging Face pour la partie embeddings du modèle.
			Args:
				hf_state (dict): État du modèle Hugging Face.
				pbar: Barre de progression optionnelle.
		"""
		prefix = "vision_tower.vision_model.embeddings."

		rename_map = {
			"patch_embedding.weight": "patch_embedding.weight",
			"patch_embedding.bias":   "patch_embedding.bias",
			"position_embedding.weight": "position_embedding.weight",
		}

		_copy_weights(self, hf_state, rename_map, prefix_src=prefix, pbar=pbar)
		print(f"[INFO] Poids importés pour {self.__class__.__name__}")
