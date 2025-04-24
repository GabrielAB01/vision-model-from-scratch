MODEL_PATH=""                     # modèle ou checkpoint HF, dossier, etc.
PROMPT="What is this monument"
IMAGE_FILE_PATH="test_images/img1.jpg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"                 # ou "True" si vous voulez échantillonner
ONLY_CPU="False"                  # "True" si pas de GPU / CUDA indisponible

python -m src.paligemma.inference \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \