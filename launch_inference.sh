MODEL_PATH="google/paligemma-3b-pt-224"                     # modèle ou checkpoint HF, dossier, etc.
PROMPT="What do you see in the picture ?"
IMAGE_FILE_PATH="test_images/img3.jpeg"
MAX_TOKENS=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"                 # ou "True" si vous voulez échantillonner
ONLY_CPU="True"                  # "True" si pas de GPU / CUDA indisponible

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

python -m paligemma.inference \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU