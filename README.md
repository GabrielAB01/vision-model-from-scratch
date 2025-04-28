# vision-model-from-scratch

> **Implémentation légère, en PyTorch, d’un modèle Vision–Langage inspiré de PaliGemma.**  
> Projet réalisé dans le cadre du cours **INF8225 – Intelligence artificielle : techniques probabilistes et d’apprentissage** (Polytechnique Montréal).
> 
> **Note** : Ce projet est une ré‑implémentation pédagogique, et non un modèle de production.


## 📑 Sommaire

- [vision-model-from-scratch](#vision-model-from-scratch)
	- [📑 Sommaire](#-sommaire)
	- [👥 Auteurs](#-auteurs)
	- [🖼️ Présentation](#️-présentation)
	- [✨ Spécifités du projet](#-spécifités-du-projet)
	- [🔗 Poids pré-entraînés](#-poids-pré-entraînés)
	- [🗂️ Arborescence](#️-arborescence)
	- [🚀 Installation en local](#-installation-en-local)
	- [Exemples d’utilisation](#exemples-dutilisation)
		- [Inférence](#inférence)
		- [Google Colab](#google-colab)
			- [**Mode d’emploi rapide (Colab) :**](#mode-demploi-rapide-colab-)
	- [Références (voir le rapport pour plus de détails)](#références-voir-le-rapport-pour-plus-de-détails)
	- [Remerciements](#remerciements)



## 👥 Auteurs

Ce projet a été réalisé par **Gabriel Abenhaim** et **Camille Yverneau**, étudiants à Polytechnique Montréal. 


## 🖼️ Présentation

Ce projet est une ré‑implémentation minimaliste mais fonctionnelle d’un **modèle Vision–Langage**, inspiré de **PaliGemma** :

* **Encodeur SigLIP‑ViT** pour l’image,
* **Décodeur Gemma** pour le texte,
* Couche de **projection multimodale** partagée, le tout en moins de **3 B paramètres**.

Le but est pédagogique : comprendre et expérimenter les briques internes d’un VLM moderne, sans dépendre de bibliothèques “magiques”.


## ✨ Spécifités du projet

| Description |                                                                                                                                      |
| :---------: | ------------------------------------------------------------------------------------------------------------------------------------ |
|      ⚙️      | **100 % PyTorch** : aucune dépendance à `transformers` pour l’inférence (seulement pour la récupération des poids, voir ci-dessous). |
|      🖼️      | **Encodeur ViT‑SigLIP** : embeddings patch + positionnels, attention multi‑têtes, MLP.                                               |
|      📝      | **Décodeur Gemma‑2B** : RoPE, self‑attention, MLP.                                                                                   |
|      🔗      | **Fusion multi‑modale** : concaténation de tokens d’image & texte, masquage automatique.                                             |
|      🏃      | **Script d’inférence prêt à l’emploi** `launch_inference.sh` : CPU/GPU, téléchargement des poids.                                    |


## 🔗 Poids pré-entraînés

Le modèle **PaliGemma** n'est **pas** ré-entraîné localement :  
il charge directement les poids publiés sur Hugging Face [paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224)


Pourquoi ce choix ?

* Le pré-entraînement complet nécessite plusieurs centaines de GPU et des semaines de calcul.
* Les fichiers `.safetensors` officiels sont publics, vérifiés et compatibles (mêmes dimensions).
* Charger ces poids permet d’évaluer, de fine-tuner ou de déployer le modèle sur des infrastructures plus légères (une carte graphique unique suffit pour l’inférence et le fine-tuning).


## 🗂️ Arborescence

```
vision-model-from-scratch/
├── TP3/                # Notebook pédagogique + mini‑traduction
├── src/                # Code source principal
│   ├── gemma/          # Décodeur texte
│   ├── paligemma/      # Pipeline vision–langage (wrapper)
│   ├── siglip/         # Encodeur image
│   └── utils/          # Fonctions partagées
├── launch_inference.sh # Script bash (download + inférence)
├── requirements.txt    # Dépendances Python
├── test.py             # Mini MLP jouet (exemple)
├── test_images/        # Images démo (hamburger, etc.)
└── README.md           # Vous êtes ici 🙂
```



## 🚀 Installation en local

> **Prérequis** : Python ≥ 3.10 ; GPU CUDA (optionnel mais recommandé).

```bash
# 1) Cloner le dépôt
$ git clone https://github.com/GabrielAB01/vision-model-from-scratch.git
$ cd vision-model-from-scratch

# 2) Installer les dépendances (PyTorch, einops, pillow…)
$ pip install -r requirements.txt  # ou poetry install
```



## Exemples d’utilisation

### Inférence

Le fichier `launch_inference.sh` permet de lancer l’inférence sur une image donnée, avec ou sans téléchargement des poids pré-entraînés.
Il est possible de spécifier :
* le modèle à utiliser ;
* le prompt d’entrée (texte) ;
* le chemin de l’image à analyser ;
* le nombre de tokens max à générer (par défaut : 128).
* la température de génération (par défaut : 0.8).
* la valeur de `top_p` (par défaut : 0.9).
* utiliser le GPU (CUDA) ou le CPU.

```bash
# Exécuter le script d’inférence (téléchargement des poids + inférence)
$ bash launch_inference.sh
```

### Google Colab


Si vous préférez essayer le modèle sans rien installer en local, un notebook Google Colab prêt à l’emploi est disponible ici :

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17KRVB0gxjdpaPjn9TtXmPoYq_Hq_c2fi#scrollTo=Obx4cDX1A3a0)

Il télécharge automatiquement les poids pré-entraînés depuis le dépôt, exécute une démonstration d’inférence sur les images d’exemple et propose un petit espace pour tester vos propres photos — le tout sur un GPU. Le fine-tuning est aussi présenté dans le notebook.

#### **Mode d’emploi rapide (Colab) :**

1. **Compressez le code** depuis votre machine :

   ```bash
   zip -r project-code.zip src test_images -x "**/__pycache__/*"
   ```

2. **Téléversez l’archive** dans votre espace Colab (`/content/`) via le panneau *Files*.

3. **Lancez les premières cellules** du notebook Colab. Elles :
   - installent les dépendances Python ;
   - décompressent l’archive ;
   - récupèrent les poids pré-entraînés.

4. **Saisissez votre token Hugging Face** lorsque la cellule le demande pour autoriser le téléchargement des poids.

> Vous pouvez ensuite exécuter les cellules d’inférence et tester le modèle sur les images fournies ou vos propres photos, sans aucune installation locale !


## Références (voir le rapport pour plus de détails)

* Beyer *et al.* 2024 – **PaliGemma** : *A Versatile 3 B VLM for Transfer*
* Zhai *et al.* 2023 – **SigLIP** : *Sigmoid Loss for Language–Image Pre‑Training*
* Dosovitskiy *et al.* 2020 – **ViT** : *An Image is Worth 16x16 Words*
* Su *et al.* 2021 – **RoFormer** : *Rotary Position Embedding*



## Remerciements

Merci au professeur **Christopher Pal** pour son accompagnement, à **Anthony Gosselin** pour le support pédagogique, et aux auteurs originaux des implémentations open‑source sur lesquelles ce projet s’appuie.

