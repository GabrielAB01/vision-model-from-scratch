from paligemma.model import PaliGemmaForConditionalGeneration
from paligemma.config import PaliGemmaConfig

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from safetensors import safe_open
from typing import Tuple

import os
import json
import glob

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
	assert tokenizer.padding_side == "right", "Tokenizer padding side must be 'right'."

	# Récupérer tous les fichiers *.safetensors dans le répertoire
	safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))

	# Charger tous les fichiers safetensors dans un dictionnaire
	tensors = {}
	for safetensors_file in safetensors_files:
		with safe_open(safetensors_file, framework="pt", device="cpu") as f:
			for key in f.keys():
				tensors[key] = f.get_tensor(key) 
	
	# Charger la configuration du modèle
	with open(os.path.join(model_dir, "config.json"), "r") as f:
		config = json.load(f)
		config = PaliGemmaConfig(**config)

	# Créer le modèle
	model = PaliGemmaForConditionalGeneration(config).to(device)

	# Charger les poids du modèle
	model.load_state_dict(tensors, strict=False)

	# Lier les poids du modèle de langage avec ceux de l'encodeur
	model.tie_weights()

	
	return model, tokenizer