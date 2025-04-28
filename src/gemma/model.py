import torch
from torch import nn
from typing import Optional, Tuple


from gemma.config import GemmaConfig
from gemma.decoder import GemmaDecoderLayer, GemmaRMSNorm
from gemma.attention import KVCache
from utils.load_weights import _copy_weights

	
class GemmaForCausalLM(nn.Module):
	"""
		Implémentation du modèle de langage Gemma pour la génération de texte.
		Permet de générer du texte à partir d'un modèle de langage.
		Contient le transformeur et la couche lineaire de sortie.
		Args:
			config (GemmaConfig): Configuration du modèle.
	"""
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.model = GemmaModel(config) # Décoder du modèle de langage
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # Pour la génération du token suivant

	def get_input_embeddings(self):
		# Renvoie les embeddings d'entrée du modèle de langage
		return self.model.embed_tokens
	
	def tie_weights(self):
		# Lie les poids de la tête de langage aux poids des embeddings d'entrée
		self.lm_head.weight = self.model.embed_tokens.weight

	def forward(
		self,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		kv_cache: Optional[KVCache] = None,
	) -> Tuple:
		"""
		Forward pass through the model.
		Args:
			attention_mask (torch.Tensor): The attention mask. [Batch_Size, Seq_Len]
			position_ids (torch.LongTensor): The position IDs. [Batch_Size, Seq_Len]
			inputs_embeds (torch.FloatTensor): The input embeddings. [Batch_Size, Seq_Len, Hidden_Size]
			kv_cache (Optional[KVCache]): The key-value cache for autoregressive decoding.
		Returns:
			Tuple: The logits and the updated key-value cache.
		"""

		# input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
		# outputs: [Batch_Size, Seq_Len, Hidden_Size]
		outputs = self.model(
			attention_mask=attention_mask,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			kv_cache=kv_cache,
		)

		hidden_states = outputs
		# [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Vocab_Size]
		logits = self.lm_head(hidden_states)
		logits = logits.float()

		return_data = {
			"logits": logits,
		}

		if kv_cache is not None:
			# Return the updated cache
			return_data["kv_cache"] = kv_cache

		return return_data
	
	def load_hf_weight(self, hf_state, pbar=None):
		# Sous-modèle (transformer)
		self.model.load_hf_weight(hf_state, pbar=pbar)
		print(f"[INFO] Poids importés pour {self.__class__.__name__}")

class GemmaModel(nn.Module):
	"""
		Implémentation du modèle Gemma.
		Applique la couche d'embedding, les couches de décodage (transformer) et la normalisation RMS.
		Args:
			config (GemmaConfig): Configuration du modèle.
	"""

	def __init__(self, config: GemmaConfig):
		super().__init__()
		self.config = config
		self.padding_idx = config.pad_token_id
		self.vocab_size = config.vocab_size

		self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
		self.layers = nn.ModuleList(
			[GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
		)
		self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def get_input_embeddings(self):
		return self.embed_tokens

	
	def forward(
		self,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		kv_cache: Optional[KVCache] = None,
	) -> torch.FloatTensor:
		
		hidden_states = inputs_embeds # [Batch_Size, Seq_Len, Hidden_Size]
		
		normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype) # [Batch_Size, Seq_Len, Hidden_Size]
		hidden_states = hidden_states * normalizer

		for decoder_layer in self.layers:
			# [Batch_Size, Seq_Len, Hidden_Size]
			hidden_states = decoder_layer(
				hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				kv_cache=kv_cache,
			)

		# [Batch_Size, Seq_Len, Hidden_Size]
		hidden_states = self.norm(hidden_states)

		# [Batch_Size, Seq_Len, Hidden_Size]
		return hidden_states
	
	def load_hf_weight(self, hf_state, pbar=None):
		# 1. embeddings
		_copy_weights(
			self.embed_tokens,
			hf_state,
			{"weight": "weight"},
			prefix_src="language_model.model.embed_tokens.",
			pbar=pbar
		)

		# 2. chaque couche du décodeur
		for idx, layer in enumerate(self.layers):
			layer.load_hf_weight(hf_state, idx, pbar=pbar)

		# 3. RMSNorm final
		self.norm.load_hf_weight(
			hf_state,
			prefix="language_model.model.norm.",
			pbar=pbar
		)
		print(f"[INFO] Poids importés pour {self.__class__.__name__}")