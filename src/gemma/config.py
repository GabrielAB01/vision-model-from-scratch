class GemmaConfig():
    """
    This is the configuration class to store the configuration of a Gemma model.
    It is used to instantiate a Gemma model according to the specified arguments.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size # Taille du vocabulaire
        self.hidden_size = hidden_size # Taille des vecteurs d'embedding
        self.intermediate_size = intermediate_size # Taille intermédiaire du réseau feedforward
        
        self.num_hidden_layers = num_hidden_layers # Nombre de couches du transformer
        self.num_attention_heads = num_attention_heads # Nombre de têtes d'attention
        self.num_key_value_heads = num_key_value_heads # Nombre de têtes d'attention pour les clés et valeurs
        
        self.head_dim = head_dim # Dimension de chaque tête d'attention
        self.max_position_embeddings = max_position_embeddings # Nombre maximum de positions d'embedding
        self.rms_norm_eps = rms_norm_eps # Epsilon pour la normalisation RMS
        self.rope_theta = rope_theta # Paramètre pour le rotary position embedding
        self.attention_bias = attention_bias # Si True, utilise un biais d'attention
        self.attention_dropout = attention_dropout # Taux de dropout pour l'attention
        self.pad_token_id = pad_token_id # ID du token de padding

