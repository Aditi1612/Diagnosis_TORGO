"""
Wav2Vec2-based Dysarthria Classifier
Uses pretrained Wav2Vec2 for speaker-invariant speech representations
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Wav2Vec2DysarthriaClassifier(nn.Module):
    """
    Wav2Vec2 encoder + classification head for dysarthria detection

    The Wav2Vec2 encoder is pretrained on large speech corpora and has learned
    speaker-invariant representations, which helps with generalization to new speakers.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
        freeze_feature_extractor: bool = True,
        pooling: str = "mean"  # "mean", "first", or "attention"
    ):
        super().__init__()

        self.pooling = pooling
        self.num_classes = num_classes

        # Load pretrained Wav2Vec2
        print(f"Loading pretrained Wav2Vec2: {model_name}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_size = self.wav2vec2.config.hidden_size  # 768 for base

        # Freeze encoder to prevent overfitting on small dataset
        if freeze_feature_extractor:
            print("Freezing Wav2Vec2 feature extractor (CNN layers)")
            self.wav2vec2.feature_extractor._freeze_parameters()

        if freeze_encoder:
            print("Freezing Wav2Vec2 transformer encoder")
            for param in self.wav2vec2.encoder.parameters():
                param.requires_grad = False

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.wav2vec2.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.wav2vec2.parameters())
        print(f"Wav2Vec2 parameters: {trainable:,} trainable / {total:,} total")

        # Classification head
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        print(f"Classification head parameters: {sum(p.numel() for p in self.classifier.parameters()):,}")

    def forward(self, input_values, attention_mask=None, return_features=False):
        """
        Forward pass

        Args:
            input_values: Raw audio waveform (batch, seq_len)
            attention_mask: Optional mask for padded sequences
            return_features: If True, also return the pooled features

        Returns:
            logits: Classification logits (batch, num_classes)
            features: (optional) Pooled features (batch, hidden_size)
        """
        # Get Wav2Vec2 hidden states
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Pool across time dimension
        if self.pooling == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)

        elif self.pooling == "first":
            pooled = hidden_states[:, 0, :]

        elif self.pooling == "attention":
            # Attention-weighted pooling
            attn_weights = self.attention(hidden_states)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            pooled = (hidden_states * attn_weights).sum(dim=1)

        # Classification
        logits = self.classifier(pooled)

        if return_features:
            return logits, pooled
        return logits


class Wav2Vec2DysarthriaClassifierLarge(Wav2Vec2DysarthriaClassifier):
    """Wav2Vec2-Large variant with 1024 hidden size"""

    def __init__(self, **kwargs):
        kwargs['model_name'] = "facebook/wav2vec2-large"
        super().__init__(**kwargs)


class HuBERTDysarthriaClassifier(Wav2Vec2DysarthriaClassifier):
    """HuBERT variant - often better for speech classification"""

    def __init__(self, **kwargs):
        kwargs['model_name'] = "facebook/hubert-base-ls960"
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Test the model
    print("Testing Wav2Vec2DysarthriaClassifier...")

    model = Wav2Vec2DysarthriaClassifier(
        freeze_encoder=True,
        freeze_feature_extractor=True,
        dropout=0.3
    )

    # Simulate batch of audio (batch_size=2, ~1 second of 16kHz audio)
    batch_size = 2
    seq_len = 16000  # 1 second at 16kHz
    dummy_audio = torch.randn(batch_size, seq_len)

    # Forward pass
    logits = model(dummy_audio)
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output: {logits}")

    # Test with features
    logits, features = model(dummy_audio, return_features=True)
    print(f"Features shape: {features.shape}")
