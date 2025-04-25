from typing import List, Tuple, Optional
import torch
from torch import nn
from einops import rearrange, repeat

from gemma.config import GemmaConfig
from utils.load_weights import _copy_weights



class GemmaAttention(nn.Module):
	def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
		super().__init__()
		self.config = config
		self.layer_idx = layer_idx

		self.attention_dropout = config.attention_dropout
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = config.head_dim
		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads
		self.max_position_embeddings = config.max_position_embeddings
		self.rope_theta = config.rope_theta
		self.is_causal = True
		self.attention_dropout = nn.Dropout(self.attention_dropout)

		assert self.hidden_size % self.num_heads == 0, "Hidden size must be divisible by number of heads"


		# Les keys et values sont projetées dans un espace de dimension [num_heads * head_dim]
		# Il y a moins de têtes d'attention pour les keys et values que pour les queries, pour accélerer le calcul (moins de transferts de données sur le GPU)
		# Grouped-query attention : pour un groupe de query heads, on utilise une seule key/value head
		self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
		self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
		self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
		self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
		

		self.rotary_emb = GemmaRotaryEmbedding(
			dim=self.head_dim,
			max_position_embeddings=self.max_position_embeddings,
			base=self.rope_theta,
		)
	
	@staticmethod
	def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
		"""
		Répète les clés et valeurs pour chaque tête de query.
		Args:
			hidden_states (torch.Tensor): Les clés ou valeurs. [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
			n_rep (int): Le nombre de répétitions.
		Returns:
			torch.Tensor: Les clés ou valeurs répétées. [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
		"""
		if n_rep == 1:
			return hidden_states
		
		return repeat(hidden_states, 'b h s d -> b (h r) s d', r=n_rep)
	
	@staticmethod
	def _apply_rotary_pos_emb(
		query_states: torch.Tensor,
		key_states: torch.Tensor,
		cos: torch.Tensor,
		sin: torch.Tensor,
		unsqueeze_dim = 1
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Applique le rotary positional embedding aux clés et valeurs.
		Args:
			query_states (torch.Tensor): Les états de la requête. [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim]
			key_states (torch.Tensor): Les états de la clé. [Batch_Size, Num_Heads_KV, Seq_Len_KV, Head_Dim]
			cos (torch.Tensor): Le cosinus de l'embedding positionnel rotatif. [Batch_Size, Seq_Len_KV, Head_Dim]
			sin (torch.Tensor): Le sinus de l'embedding positionnel rotatif. [Batch_Size, Seq_Len_KV, Head_Dim]
		Returns:
			Tuple[torch.Tensor, torch.Tensor]: Les clés et valeurs avec l'embedding positionnel appliqué.
		"""
		# Ajout de la head dimension
		cos = cos.unsqueeze(unsqueeze_dim)
		sin = sin.unsqueeze(unsqueeze_dim)

		# On applique le rotary positional embedding
		query_embed = (query_states * cos) + (GemmaAttention._rotate_half(query_states) * sin)
		key_embed = (key_states * cos) + (GemmaAttention._rotate_half(key_states) * sin)
		return query_embed, key_embed

	@staticmethod
	def _rotate_half(x: torch.Tensor) -> torch.Tensor:
		"""
		Effectue une rotation de moitié sur les clés et valeurs.
		Permet de construire le vecteur [-x_{d//2} ... -x_{d-1}, x_0 ... x_{d//2 - 1}]
		L'implémentation diffère de celle de l'originale, dû à des permutation effectuées à d'autres endroits.
		Mais le résultat final est le même.
		Args:
			x (torch.Tensor): Les clés ou valeurs. [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
		Returns:
			torch.Tensor: Les clés ou valeurs après rotation. [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
		"""
		x1 = x[..., :x.size(-1) // 2] # Récupère la première moitié sur la dernière dimension
		x2 = x[..., x.size(-1) // 2:] # Récupère la deuxième moitié sur la dernière dimension
		return torch.cat((-x2, x1), dim=-1)

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		kv_cache: Optional['KVCache'] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		
		batch_size, seq_len, _ = hidden_states.size()
	
		# -------- Projection des clés, valeurs et queries --------
		query_states = self.q_proj(hidden_states) # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
		query_states = rearrange(query_states, 'b s (h d) -> b h s d', h=self.num_heads) # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]

		key_states = self.k_proj(hidden_states) # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
		key_states = rearrange(key_states, 'b s (h d) -> b h s d', h=self.num_key_value_heads) # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]

		value_states = self.v_proj(hidden_states) # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
		value_states = rearrange(value_states, 'b s (h d) -> b h s d', h=self.num_key_value_heads) # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]

		# --------- Rotary Positional Embedding ---------
		# [Batch_Size, Seq_Len, Head_Dim]
		cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

		query_states, key_states = GemmaAttention._apply_rotary_pos_emb(query_states, key_states, cos, sin)

		if kv_cache is not None:
			# On concatène les nouvelles clés et valeurs avec les anciennes
			key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
			
		# Répétition des clés et valeurs pour match le nombre de têtes de la query
		key_states = GemmaAttention._repeat_kv(key_states, self.num_key_value_groups) # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
		value_states = GemmaAttention._repeat_kv(value_states, self.num_key_value_groups) # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]

		# --------- Attention ---------
		attn_weights = torch.einsum('bhqd,bhkd->bhqk', query_states, key_states) / (self.head_dim**0.5) # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]

		assert attention_mask is not None, "GemmaAttention: attention_mask must be provided"
		# On applique le masque d'attention
		attn_weights = attn_weights + attention_mask

		# -------- Softmax et Dropout --------- 
		attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
		attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout.p, training=self.training)

		# --------- Multiplication par les valeurs ---------
		attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value_states) # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim]

		if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
			raise ValueError(
				f"Attention output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, "
				f"but got {attn_output.size()}"
			)
		# --------- Projection de sortie ---------
		attn_output = rearrange(attn_output, 'b h q d -> b q (h d)')
		attn_output = self.o_proj(attn_output) # [Batch_Size, Seq_Len, Hidden_Size]

		
		return attn_output, attn_weights
	
	def load_hf_weight(self, hf_state: dict, layer_idx: int):
		"""
			Args
			----
			hf_state  : state-dict Hugging Face déjà en mémoire
			layer_idx : index de la couche dans le modèle HF
		"""
		# Préfixe exact des clés Hugging Face pour cette couche
		prefix = f"language_model.model.layers.{layer_idx}.self_attn."

		# Table de renommage 1-pour-1 (clé HF relative -> clé locale)
		rename_map = {
			"q_proj.weight": "q_proj.weight",
			"k_proj.weight": "k_proj.weight",
			"v_proj.weight": "v_proj.weight",
			"o_proj.weight": "o_proj.weight",
		}

		# Les biais n’existent que si `attention_bias=True`
		if self.config.attention_bias:
			rename_map.update({
				"q_proj.bias": "q_proj.bias",
				"k_proj.bias": "k_proj.bias",
				"v_proj.bias": "v_proj.bias",
				"o_proj.bias": "o_proj.bias",
			})

		# Copie effective
		_copy_weights(self, hf_state, rename_map, prefix_src=prefix)



class KVCache():
	"""
	KV-Cache pour les clés et valeurs de l'attention.
	Contient les clés et valeurs de chaque couche du décodeur.
	"""
	def __init__(self) -> None:
		self.key_cache: List[torch.Tensor] = []
		self.value_cache: List[torch.Tensor] = []
	
	def num_items(self) -> int:
		if len(self.key_cache) == 0:
			return 0
		else:
			# The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
			return self.key_cache[0].shape[-2]

	def update(
		self,
		key_states: torch.Tensor,
		value_states: torch.Tensor,
		layer_idx: int,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Met à jour le KV-Cache avec les nouvelles clés et valeurs.
		Args:
			key_states (torch.Tensor): Les nouvelles clés. [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
			value_states (torch.Tensor): Les nouvelles valeurs. [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
			layer_idx (int): L'index de la couche.
		Returns:
			Tuple[torch.Tensor, torch.Tensor]: Les clés et valeurs mises à jour.
		"""
		if len(self.key_cache) <= layer_idx:
			# Si on n'a pas encore de cache pour cette couche, on l'initialise.
			self.key_cache.append(key_states)
			self.value_cache.append(value_states)
		else:
			# Sinon on concatène les nouvelles clés et valeurs avec les anciennes
			# [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim] -> [Batch_Size, Num_Heads_KV, Seq_Len + Seq_Len, Head_Dim]
			self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
			self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

		# ... et ensuite on retourne toutes les clés existantes + les nouvelles.
		return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaRotaryEmbedding(nn.Module):
	"""
	Implémentation du rotary positional embedding.
	Args:
		dim (int): La dimension de la tête.
		max_position_embeddings (int): Le nombre maximum d'embeddings.
		base (float): La base pour le calcul de l'embedding.
	"""
	def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
		super().__init__()
		self.dim = dim
		self.max_position_embeddings = max_position_embeddings
		self.base = base

		# Calculer les thetas avec la formule theta_i = base^(-2i/dim) pour i = 0, 2, 4, ..., dim-2
		inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / self.dim))
		self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

	@torch.no_grad()
	def forward(self, x: torch.Tensor, position_ids: torch.LongTensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Applique le rotary positional embedding.
		Args:
			x (torch.Tensor): Les clés ou valeurs. [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
			position_ids (torch.LongTensor): Les IDs de position. [Batch_Size, Seq_Len]
			seq_len (Optional[int]): La longueur de la séquence.
		Returns:
			Tuple[torch.Tensor, torch.Tensor]: Le cosinus et le sinus de l'embedding positionnel rotatif --> [Batch_Size, Seq_Len, Head_Dim]
		"""
		inv_freq = self.inv_freq.to(x.device)

		# On duplique le buffer le long de la séquence : [Head_Dim // 2] -> [Batch_Size, Head_Dim // 2, 1]
		inv_freq_expanded  = inv_freq[None, :, None].float().expand(position_ids.size(0), -1, 1)
		# Pareil pour les positions : [Batch_Size, Seq_Len] -> [Batch_Size, 1, Seq_Len]
		position_ids_expanded = position_ids[:, None, :].float()

		device_type = x.device
		device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'

		# Calculer les positions
		with torch.autocast(device_type=device_type, enabled=False):
			# Multiplier chaque theta par la position
			# [Batch_Size, Head_Dim//2, 1] @ [Batch_Size, 1, Seq_Len] -> [Batch_Size, Seq_Len, Head_Dim//2]
			freqs = torch.einsum('bdr,brs->bsd', inv_freq_expanded.float(), position_ids_expanded.float())
			# Same as : freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

			emb = torch.cat((freqs, freqs), dim=-1) # [Batch_Size, Seq_Len, Head_Dim]

			cos = emb.cos()
			sin = emb.sin()

		return cos.to(x.dtype), sin.to(x.dtype)