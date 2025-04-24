import torch
from torch import nn
from typing import Optional, Tuple
from einops import rearrange, repeat

from siglip.model import SiglipVisionModel
from paligemma.config import PaliGemmaConfig

from gemma.attention import KVCache
from gemma.model import GemmaForCausalLM


class PaliGemmaForConditionalGeneration(nn.Module):
	"""
		Implémentation du modèle PaliGemma pour la génération conditionnelle.
		Args:
			config (PaliGemmaConfig): Configuration du modèle.
	"""
	def __init__(self, config: PaliGemmaConfig):
		super().__init__()
		self.config = config
		self.vision_tower = SiglipVisionModel(config.vision_config)
		self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
		self.vocab_size = config.vocab_size

		language_model = GemmaForCausalLM(config.text_config)
		self.language_model = language_model

		self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

	def tie_weights(self):
		"""
		Permet de lier les poids du modèle de langage avec ceux de l'encodeur.
		"""
		return self.language_model.tie_weights()
	
	def _merge_input_ids_with_image_features(
		self,
		image_features: torch.Tensor,
		inputs_embeds: torch.Tensor,
		input_ids: torch.LongTensor,
		attention_mask: torch.Tensor,
		kv_cache: Optional[KVCache] = None
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Fusionne les identifiants d'entrée avec les caractéristiques d'image pour créer un tenseur d'embedding final, 
		ainsi qu'un masque d'attention causale et des identifiants de position pour une utilisation dans le modèle de langage.

		Arguments:
			image_features (torch.Tensor): Un tenseur de forme [Batch_Size, Num_Image_Tokens, Hidden_Size] contenant 
				les caractéristiques d'image extraites d'un encodeur d'image.
			inputs_embeds (torch.Tensor): Un tenseur de forme [Batch_Size, Seq_Len, Hidden_Size] contenant les embeddings 
				des tokens d'entrée.
			input_ids (torch.LongTensor): Un tenseur de forme [Batch_Size, Seq_Len] contenant les identifiants des tokens 
				d'entrée.
			attention_mask (torch.Tensor): Un tenseur de forme [Batch_Size, Seq_Len] indiquant quels tokens doivent être 
				pris en compte (1 pour les tokens valides, 0 pour le padding).
			kv_cache (Optional[KVCache]): Un objet cache clé-valeur optionnel utilisé pour un décodage autoregressif 
				efficace. Si fourni, il contient des paires clé-valeur calculées précédemment.

		Retourne:
				- final_embedding (torch.Tensor): Un tenseur de forme [Batch_Size, Seq_Len, Hidden_Size] contenant les 
				  embeddings fusionnés des tokens de texte, d'image et de padding.
				- causal_mask (torch.Tensor): Un tenseur de forme [Batch_Size, Num_Heads_Q, q_len, kv_len] représentant 
				  le masque d'attention causale pour le transformeur.
				- position_ids (torch.Tensor): Un tenseur de forme [Batch_Size, Seq_Len] contenant les identifiants de 
				  position pour les tokens d'entrée.

			- La méthode combine les embeddings de texte, les embeddings d'image (mis à l'échelle par la taille cachée) 
			  et les tokens de padding dans un seul tenseur d'embedding.
			- Des masques sont créés pour distinguer les tokens de texte, d'image et de padding, et sont utilisés pour 
			  mettre à jour sélectivement le tenseur d'embedding final.
			- Le masque d'attention causale est construit différemment selon que le `kv_cache` est fourni (indiquant un 
			  décodage autoregressif) ou non (indiquant un mode de pré-remplissage).
			- Les identifiants de position sont calculés en fonction du masque d'attention, avec un traitement spécial 
			  pour le décodage autoregressif.
		"""
		_, _, embed_dim = image_features.shape
		batch_size, seq_len = input_ids.shape
		dtype, device = inputs_embeds.dtype, inputs_embeds.device

		# [Batch_Size, Seq_Len, Hidden_Size]
		scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

		# Combiner les embeddings de l'image, du texte, et ajouter un masque sur les tokens de padding
		final_embedding = torch.zeros((batch_size, seq_len, embed_dim), dtype=dtype, device=device)

		# [Batch_Size, Seq_Len] : True pour les tokens de texte
		text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
		# [Batch_Size, Seq_Len] : True pour les tokens d'image
		image_mask = (input_ids == self.config.image_token_index)
		# [Batch_Size, Seq_Len] : True pour les tokens de padding
		padding_mask = (input_ids == self.pad_token_id)

		# Etendre les masques pour correspondre à la taille des embeddings
		# text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
		text_mask = repeat(text_mask, 'b s -> b s d', d=embed_dim)
		image_mask = repeat(image_mask, 'b s -> b s d', d=embed_dim)
		padding_mask = repeat(padding_mask, 'b s -> b s d', d=embed_dim)

		# Ajouter les embeddings de texte
		final_embedding = torch.where(text_mask, inputs_embeds, final_embedding)
		# Ajouter les embeddings d'image
		# On ne peut pas utiliser `torch.where` ici car les dimensions ne correspondent pas
		# La longueur de scaled_image_features n'est pas égale à la longueur de l'embedding final
		final_embedding = final_embedding.masked_scatter(image_mask, scaled_image_features)
		# Ajouter les embeddings de padding
		final_embedding = torch.where(padding_mask, torch.zeros_like(final_embedding), final_embedding)


		# ------ Créer le masque d'attention ------

		q_len = inputs_embeds.shape[1]

		if kv_cache is None or kv_cache.num_items() == 0:
			# On ne masque aucun token, car on est en phase de 'prefill'
			causal_mask = torch.full(
				(batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
			)
		else:
			# Comme on génère des tokens, la query doit avoir un seul token
			assert q_len == 1, "The query length must be 1 when using kv_cache"
			kv_len = kv_cache.num_items() + q_len

			causal_mask = torch.full(
				(batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
			)

		# Ajouter la dimension d'attention
		# [Batch_Size, q_len, kv_len] -> [Batch_Size, Num_Heads_Q, q_len, kv_len]
		causal_mask = rearrange(causal_mask, 'b q k -> b 1 q k')

		if kv_cache is not None and kv_cache.num_items() > 0:
			# La postion de la clé est juste la dernière position de la séquence
			position_ids = attention_mask.cumsum(-1)[:, -1]
			if position_ids.dim() == 1:
				position_ids = position_ids.unsqueeze(0)
		else:
			# Calcule la somme cumulative du masque d'attention le long de la dernière dimension, définit les positions où le masque d'attention est 0 à 1
			position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

		return final_embedding, causal_mask, position_ids

		
	def forward(
		self,
		input_ids: torch.LongTensor,
		pixel_values: torch.FloatTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		kv_cache: Optional[KVCache] = None
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Fait passer les entrées à travers le modèle.
		Args:
			input_ids (torch.Tensor): Les identifiants d'entrée.
			attention_mask (Optional[torch.Tensor]): Le masque d'attention.
			pixel_values (Optional[torch.Tensor]): Les valeurs des pixels.
			pixel_mask (Optional[torch.Tensor]): Le masque des pixels.
			return_dict (bool): Si True, retourne un dictionnaire. Sinon, retourne un tuple.
		Returns:
			Tuple[torch.Tensor, torch.Tensor]: Les logits et les valeurs de perte.
		"""
		# Make sure the input is right-padded
		assert torch.all(attention_mask == 1), "The input cannot be padded"
		
		# 1. Extract the input embeddings
		# shape: (Batch_Size, Seq_Len, Hidden_Size)
		inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

		# 2. Merge text and images
		# [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
		selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype)) # Result of SiglipVisionModel
		# [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
		image_features = self.multi_modal_projector(selected_image_feature)

		# Merge the embeddings of the text tokens and the image tokens
		inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)
		
		outputs = self.language_model(
			attention_mask=attention_mask,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			kv_cache=kv_cache,
		)

		return outputs

class PaliGemmaMultiModalProjector(nn.Module):
	"""
		Implémentation du projecteur multi-modal pour le modèle PaliGemma.
		Réalise simplement une projection linéaire des caractéristiques d'image.
		Args:
			config (PaliGemmaConfig): Configuration du modèle.
	"""
	def __init__(self, config: PaliGemmaConfig):
		super().__init__()
		self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

	def forward(self, image_features):
		"""
			Forward pass à travers le projecteur multi-modal.
			Args:
				image_features (torch.Tensor): Caractéristiques d'image de forme [Batch_Size, Num_Patches, Hidden_Size].
			Returns:
				torch.Tensor: Caractéristiques d'image projetées de forme [Batch_Size, Num_Patches, Projection_Dim].
		"""
		# [Batch_Size, Num_Patches, Hidden_Size] -> [Batch_Size, Num_Patches, Projection_Dim]
		hidden_states = self.linear(image_features)
		return hidden_states