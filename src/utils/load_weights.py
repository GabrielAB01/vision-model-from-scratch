from collections import OrderedDict
import torch.nn as nn

# ----------------------------------------------------------------------------- #
# Utilitaire minimal : copie (et renomme) les tenseurs d’un state-dict HF vers  #
# un module PyTorch.                                                            #
# ----------------------------------------------------------------------------- #
def _copy_weights(module: nn.Module, hf_state: dict, rename_map: dict, prefix_src=""):
    """
        module      : nn.Module local à remplir
        hf_state    : dict[str, Tensor] - state-dict Hugging Face déjà chargé
        rename_map  : dict[str, str]    - {'clé_relative_HF': 'clé_locale'}
        prefix_src  : préfixe Hugging Face à écarter (ex : "…layers.0.self_attn.")
    """
    renamed = OrderedDict()
    for old_key, new_key in rename_map.items():
        full_hf_key = prefix_src + old_key
        if full_hf_key not in hf_state:
            print(f"[load_hf] clé manquante : {full_hf_key} - {old_key} -> {new_key}")
            continue
        renamed[new_key] = hf_state[full_hf_key]

    # strict=False = on laisse PyTorch signaler ce qu’il manque / en trop
    missing, unexpected = module.load_state_dict(renamed, strict=False)
    if missing:
        print(f"[load_hf] paramètres non fournis : {missing} - {prefix_src}")
    if unexpected:
        print(f"[load_hf] paramètres inconnus : {unexpected} - {prefix_src}")
