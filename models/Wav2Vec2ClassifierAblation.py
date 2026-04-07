"""
Wav2Vec2-based Dysarthria Classifier with Extended Features for Ablation Studies
Supports: weighted-sum pooling, layer-wise analysis, max pooling
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel


class Wav2Vec2AblationClassifier(nn.Module):
    """
    Extended Wav2Vec2 classifier for ablation studies

    Additional features:
    - Weighted-sum pooling across layers
    - Layer-wise feature extraction
    - Max pooling option
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        freeze_feature_extractor: bool = True,
        pooling: str = "mean",  # "mean", "first", "attention", "max", "weighted_sum"
        use_layer: int = -1,  # -1 for last layer, 0-11 for specific layer
        model_type: str = "wav2vec2"  # "wav2vec2" or "hubert"
    ):
        super().__init__()

        self.pooling = pooling
        self.num_classes = num_classes
        self.use_layer = use_layer
        self.model_type = model_type

        # Load pretrained model
        print(f"Loading pretrained model: {model_name}")
        if model_type == "hubert":
            self.encoder = HubertModel.from_pretrained(model_name)
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        self.hidden_size = self.encoder.config.hidden_size
        self.num_layers = self.encoder.config.num_hidden_layers

        # Freeze feature extractor (CNN layers)
        if freeze_feature_extractor:
            print("Freezing feature extractor (CNN layers)")
            self.encoder.feature_extractor._freeze_parameters()

        # Freeze transformer encoder
        if freeze_encoder:
            print("Freezing transformer encoder")
            for param in self.encoder.encoder.parameters():
                param.requires_grad = False

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        print(f"Encoder parameters: {trainable:,} trainable / {total:,} total")

        # Weighted-sum layer weights (learnable)
        if pooling == "weighted_sum":
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers + 1) / (self.num_layers + 1))
            print(f"Using weighted-sum pooling across {self.num_layers + 1} layers")

        # Attention pooling
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        print(f"Pooling strategy: {pooling}")
        if use_layer >= 0:
            print(f"Using layer {use_layer} for features")
        print(f"Classifier parameters: {sum(p.numel() for p in self.classifier.parameters()):,}")

    def forward(self, input_values, attention_mask=None, return_features=False, return_all_layers=False):
        """
        Forward pass

        Args:
            input_values: Raw audio waveform (batch, seq_len)
            attention_mask: Optional mask for padded sequences
            return_features: If True, return pooled features
            return_all_layers: If True, return all layer hidden states
        """
        # Get all hidden states
        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        all_hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors

        # Select which hidden state(s) to use
        if self.pooling == "weighted_sum":
            # Weighted sum across all layers
            stacked = torch.stack(all_hidden_states, dim=0)  # (num_layers+1, batch, seq, hidden)
            weights = torch.softmax(self.layer_weights, dim=0)
            weights = weights.view(-1, 1, 1, 1)
            hidden_states = (stacked * weights).sum(dim=0)  # (batch, seq, hidden)
        elif self.use_layer >= 0:
            # Use specific layer
            hidden_states = all_hidden_states[self.use_layer]
        else:
            # Use last layer
            hidden_states = outputs.last_hidden_state

        # Pool across time dimension
        if self.pooling == "mean" or self.pooling == "weighted_sum":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)

        elif self.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                hidden_states = hidden_states.masked_fill(mask == 0, float('-inf'))
            pooled = hidden_states.max(dim=1)[0]

        elif self.pooling == "first":
            pooled = hidden_states[:, 0, :]

        elif self.pooling == "attention":
            attn_weights = self.attention(hidden_states)
            attn_weights = torch.softmax(attn_weights, dim=1)
            pooled = (hidden_states * attn_weights).sum(dim=1)

        # Classification
        logits = self.classifier(pooled)

        if return_all_layers:
            return logits, pooled, all_hidden_states
        if return_features:
            return logits, pooled
        return logits


class LayerWiseClassifier(nn.Module):
    """
    Classifier that extracts features from a specific layer for layer-wise analysis
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        dropout: float = 0.3,
        layer_index: int = -1,  # Which layer to use (-1 = last)
        model_type: str = "wav2vec2"
    ):
        super().__init__()

        self.layer_index = layer_index
        self.model_type = model_type

        # Load pretrained model
        if model_type == "hubert":
            self.encoder = HubertModel.from_pretrained(model_name)
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        self.hidden_size = self.encoder.config.hidden_size
        self.num_layers = self.encoder.config.num_hidden_layers

        # Freeze all encoder parameters for probing
        self.encoder.feature_extractor._freeze_parameters()
        for param in self.encoder.encoder.parameters():
            param.requires_grad = False

        print(f"Layer-wise classifier using layer {layer_index}")
        print(f"All encoder parameters frozen for probing")

        # Simple linear classifier for probing
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes)
        )

    def forward(self, input_values, attention_mask=None):
        with torch.no_grad():
            outputs = self.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Get specific layer
        if self.layer_index == -1:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[self.layer_index]

        # Mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        return self.classifier(pooled)


if __name__ == "__main__":
    print("Testing Wav2Vec2AblationClassifier...")

    # Test weighted-sum pooling
    model = Wav2Vec2AblationClassifier(
        pooling="weighted_sum",
        freeze_encoder=False
    )

    dummy_audio = torch.randn(2, 16000)
    logits = model(dummy_audio)
    print(f"Weighted-sum output shape: {logits.shape}")

    # Test layer-wise classifier
    print("\nTesting LayerWiseClassifier...")
    for layer in [0, 6, 11]:
        model = LayerWiseClassifier(layer_index=layer)
        logits = model(dummy_audio)
        print(f"Layer {layer} output shape: {logits.shape}")
