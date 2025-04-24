import sys
import os


from PIL import Image
import torch
import fire

from paligemma.load_model import load_hf_model
from paligemma.model import PaliGemmaForConditionalGeneration
from gemma.attention import KVCache
from paligemma.preprocessing import PaliGemmaProcessor


# Déplace les entrées du modèle vers le device spécifié
def move_inputs_to_device(model_inputs: dict, device: str):
	model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
	return model_inputs

# Récupère les entrées du modèle à partir du processeur et de l'image
def get_model_inputs(processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str):
	# Load image and prompt
	images = [Image.open(image_file_path)]
	prompts = [prompt]

	model_inputs = processor(text=prompts, images=images)
	model_inputs = move_inputs_to_device(model_inputs, device)
	return model_inputs


# Fonction de sampling pour le top-p sampling
def _sample_top_p_autre(probs: torch.Tensor, top_p: float):

	# Trier les probabilités dans l'ordre décroissant
	# sorted_probs, sorted_indices : [Batch_Size, Vocab_Size]
	sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
	cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # Calculer la somme cumulative des probabilités


	threshold_index = (cumulative_probs <= top_p).sum(dim=-1) - 1
	threshold_index = torch.clamp(threshold_index, min=0)


	mask = torch.zeros_like(probs, dtype=torch.bool)
	mask.scatter_(1, sorted_indices[:, :threshold_index.unsqueeze(1)], True)


	adjusted_probs = probs.masked_fill(~mask, 0.0)
	adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)

	return torch.multinomial(adjusted_probs, num_samples=1)

def _sample_top_p(probs: torch.Tensor, p: float):
	# Trier les probabilités dans l'ordre décroissant
	probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # [Batch_Size, Vocab_Size]
	# Calculer la somme cumulative des probabilités
	probs_sum = torch.cumsum(probs_sort, dim=-1)
	
	# Trouver le premier index où la somme cumulative dépasse p
	mask = probs_sum - probs_sort > p
	# Mettre à zéro les probabilités au-delà de ce point
	probs_sort[mask] = 0.0
	# Normaliser les probabilités restantes
	probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

	# Échantillonner un index à partir des probabilités restantes
	# On utilise torch.multinomial pour échantillonner un index à partir des probabilités
	next_token = torch.multinomial(probs_sort, num_samples=1)
	# Obtenir la position du token dans le vocabulaire correspondant à l'index échantillonné
	next_token = torch.gather(probs_idx, -1, next_token)
	return next_token





def test_inference(
		model: PaliGemmaForConditionalGeneration,
		processor: PaliGemmaProcessor,
		device: str,
		prompt: str,
		image_file_path: str,
		max_tokens: int,
		temperature: float,
		top_p: float,
		do_sample: bool,
):
	# On récupère les entrées du modèle à partir du prompt et de l'image
	model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
	input_ids = model_inputs["input_ids"]
	attention_mask = model_inputs["attention_mask"]
	pixel_values = model_inputs["pixel_values"]
	
	# Initialisation du KV-Cache
	kv_cache = KVCache()


	# ----------- Génération de texte -----------
	stop_token = processor.tokenizer.eos_token_id
	generated_tokens = []

	for _ in range(max_tokens):
		outputs = model(
			input_ids=input_ids,
			pixel_values=pixel_values,
			attention_mask=attention_mask,
			kv_cache=kv_cache,
		)
		# Mettre à jour le cache
		kv_cache = outputs["kv_cache"]

		# Outputs["logits"] : [Batch_Size, Seq_Len, Vocab_Size]
		# On ne garde que le dernier token : [Batch_Size, 1, Vocab_Size]
		next_token_logits = outputs["logits"][:, -1, :] 

		if do_sample:
			# Appliquer la température et le top-p sampling
			next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
			next_token = _sample_top_p(next_token_logits, top_p)
		else:
			next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

		assert next_token.shape == (1, 1), f"next_token shape is {next_token.shape}, expected (1, 1)"

		next_token = next_token.squeeze(0) # Enleve la dimension de batch
		generated_tokens.append(next_token)

		# On arrête la génération si on atteint le token d'arrêt
		if next_token.item() == stop_token:
			break

		# On met à jour les entrées du modèle
		input_ids = next_token.unsqueeze(-1)
		attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1)

	# On convertit les tokens générés en texte
	generated_tokens = torch.cat(generated_tokens, dim=-1)
	generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

	print(f"{prompt} : \n{generated_text}")
	return generated_text


def main(
	model_path: str = None,
	prompt: str = None,
	image_file_path: str = None,
	max_tokens: int = 100,
	temperature: float = 0.8,
	top_p: float = 0.9,
	do_sample: bool = False,
	only_cpu: bool = False,
):
	"""
	Function to run the inference.
	Args:
		model_path (str): Path to the model.
		prompt (str): The prompt to use for the inference.
		image_file_path (str): Path to the image file.
		max_tokens (int): Maximum number of tokens to generate.
		temperature (float): Temperature for sampling.
		top_p (float): Top-p sampling parameter.
		do_sample (bool): Whether to use sampling or not.
		only_cpu (bool): Whether to use only CPU or not.
	"""
	if only_cpu:
		device = "cpu"
	else:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	print(f"Using device: {device}")

	print(f"Loading model from {model_path}...")
	model, tokenizer = load_hf_model(model_path, device)
	model = model.to(device).eval()

	print("Loading processor...")
	processor = PaliGemmaProcessor.from_pretrained(model_path)

	num_image_tokens = model.config.vision_config.num_image_tokens
	image_size = model.config.vision_config.image_size
	processor = PaliGemmaProcessor(
		tokenizer=tokenizer,
		num_image_tokens=num_image_tokens,
		image_size=image_size,
	)

	print("Running inference...")
	with torch.no_grad():
		test_inference(
			model,
			processor,
			device,
			prompt,
			image_file_path,
			max_tokens,
			temperature,
			top_p,
			do_sample,
		)


if __name__ == "__main__":
	# Exécutez le script avec les arguments en ligne de commande
	fire.Fire(main)