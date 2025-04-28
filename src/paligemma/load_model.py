from paligemma.model import PaliGemmaForConditionalGeneration
from paligemma.config import PaliGemmaConfig

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from safetensors import safe_open
from tqdm.auto import tqdm

import os
import json
import glob
import torch

def _ensure_local(model_path: str):
	"""Si `model_path` est un repo HF, télécharge-le et renvoie le dossier local."""
	if os.path.isdir(model_path):
		return model_path
	# Sinon on suppose que c'est un repo_id Hugging Face
	return snapshot_download(
		repo_id=model_path,
		allow_patterns=["*.safetensors", "*.json"],
		resume_download=True,
	)


def load_hf_model(
	model_path: str,
	device: str = "cuda",
):
	model_dir = _ensure_local(model_path) # Vérifie si le modèle est local ou un repo Hugging Face
	# Charger le tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")

	# 1) Charger les poids safetensors, s’il y en a
	tensors = {}
	for f in glob.glob(os.path.join(model_dir, "*.safetensors")):
			with safe_open(f, framework="pt", device="cpu") as sf:
					for k in sf.keys():
							tensors[k] = sf.get_tensor(k)

	# 2) Charger le pytorch_model.bin s’il existe
	bin_path = os.path.join(model_dir, "pytorch_model.bin")
	if os.path.exists(bin_path):
			bin_tensors = torch.load(bin_path, map_location="cpu")
			tensors.update(bin_tensors)
	
	# Charger la configuration du modèle
	with open(os.path.join(model_dir, "config.json"), "r") as f:
		cfg_dict = json.load(f)
	# Éviter de passer deux fois pad_token_id à GemmaConfig
	if 'text_config' in cfg_dict and 'pad_token_id' in cfg_dict['text_config']:
		cfg_dict['text_config'].pop('pad_token_id')
	config = PaliGemmaConfig(**cfg_dict)

	# Créer le modèle
	model = PaliGemmaForConditionalGeneration(config).to(device)

	# Charger les poids du modèle
	# model.load_state_dict(tensors, strict=False)
	total_keys = len(tensors)               # ≈ nombre de tensors à copier
	with tqdm(total=total_keys,
			  desc="Loading weights",
			  unit=" tensors",
			) as pbar:
		model.load_hf_weight(tensors, pbar=pbar)

	# Lier les poids du modèle de langage avec ceux de l'encodeur
	model.tie_weights()

	
	return model, tokenizer