"""
-*- coding: utf-8 -*-
DeepThink-Speech: BiLSTM with Multi-Head Attention for Dysarthria Detection
Integrates with existing attention mechanisms for explainable AI
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .base import BaseNet
    from .transformer_timm import Attention
except ImportError:
    # If relative import fails, use absolute import
    sys.path.insert(0, str(Path(__file__).parent))
    from base import BaseNet
    from transformer_timm import Attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module that captures attention weights for explainability"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store for explainability

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights.detach()  # Store for explainability
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.out_linear(context)

        return output, self.attention_weights


class BiLSTM_MHA_Dysarthria(BaseNet):
    """
    BiLSTM with Multi-Head Attention for Dysarthria Detection
    Stage 1 of DeepThink-Speech Framework

    Architecture:
    1. Input: MFCC features (time_steps, n_mfcc)
    2. BiLSTM layers for temporal modeling
    3. Multi-Head Attention for capturing discriminative regions
    4. Classification head
    """

    def __init__(self,
                 input_dim=40,           # MFCC features (typically 13-40)
                 hidden_dim=256,          # LSTM hidden dimension
                 num_layers=2,            # Number of BiLSTM layers
                 num_heads=8,             # Number of attention heads
                 dropout=0.3,
                 num_classes=2,           # Binary: dysarthric vs healthy
                 attention_dim=128):      # Attention output dimension
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            d_model=hidden_dim * 2,  # *2 for bidirectional
            num_heads=num_heads,
            dropout=dropout
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.fc1 = nn.Linear(hidden_dim * 2, attention_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(attention_dim, num_classes)

        # Attention weights storage for explainability
        self.last_attention_weights = None
        self.last_lstm_output = None

    def feature_extractor(self, x, return_attention=False):
        """
        Extract features from MFCC input

        Args:
            x: Input tensor of shape (batch, time_steps, n_mfcc)
            return_attention: If True, return attention weights for explainability

        Returns:
            features: Extracted features
            attention_weights: (optional) Attention weights for explainability
        """
        # Input normalization
        x_norm = self.input_norm(x.transpose(1, 2)).transpose(1, 2)

        # BiLSTM forward pass
        lstm_out, (h_n, c_n) = self.bilstm(x_norm)
        self.last_lstm_output = lstm_out.detach()  # Store for acoustic feature extraction

        # Multi-Head Attention
        attn_out, attention_weights = self.attention(lstm_out)
        self.last_attention_weights = attention_weights

        # Residual connection and layer norm
        lstm_out = self.layer_norm(lstm_out + attn_out)

        # Global average pooling over time dimension
        pooled = torch.mean(lstm_out, dim=1)

        # Dropout
        features = self.dropout(pooled)

        if return_attention:
            return features, attention_weights
        return features

    def classifier(self, features):
        """Classification head"""
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def forward(self, x, return_attention=False):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, time_steps, n_mfcc)
            return_attention: If True, return attention weights

        Returns:
            logits: Classification logits
            attention_weights: (optional) Attention weights
        """
        if return_attention:
            features, attention_weights = self.feature_extractor(x, return_attention=True)
            logits = self.classifier(features)
            return logits, attention_weights
        else:
            features = self.feature_extractor(x, return_attention=False)
            logits = self.classifier(features)
            return logits

    def get_attention_weights(self):
        """Get the last computed attention weights for explainability"""
        return self.last_attention_weights

    def get_high_attention_regions(self, threshold_percentile=75):
        """
        Identify temporal regions with high attention for acoustic feature extraction

        Args:
            threshold_percentile: Percentile threshold for high attention (default 75)

        Returns:
            high_attention_mask: Boolean mask of high-attention time steps
            attention_scores: Aggregated attention scores per time step
        """
        if self.last_attention_weights is None:
            raise ValueError("No attention weights available. Run forward pass first.")

        # Average attention weights across heads and query positions
        # Shape: (batch, num_heads, seq_len, seq_len) -> (batch, seq_len)
        attention_scores = self.last_attention_weights.mean(dim=1).mean(dim=1)

        # Compute threshold based on percentile
        batch_size = attention_scores.shape[0]
        high_attention_mask = torch.zeros_like(attention_scores, dtype=torch.bool)

        for i in range(batch_size):
            threshold = torch.quantile(attention_scores[i], threshold_percentile / 100.0)
            high_attention_mask[i] = attention_scores[i] >= threshold

        return high_attention_mask, attention_scores


class HybridExplainableModel(nn.Module):
    """
    Hybrid Explainable AI Model combining BiLSTM-MHA with existing attention mechanisms
    Can leverage both BiLSTM attention and cross-modal attention from MultiModalDepDet
    """

    def __init__(self,
                 bilstm_config=None,
                 use_cross_modal_attention=False):
        super().__init__()

        # BiLSTM-MHA for speech
        if bilstm_config is None:
            bilstm_config = {
                'input_dim': 40,
                'hidden_dim': 256,
                'num_layers': 2,
                'num_heads': 8,
                'dropout': 0.3,
                'num_classes': 2
            }

        self.speech_model = BiLSTM_MHA_Dysarthria(**bilstm_config)

        # Optional: Cross-modal attention (can integrate with MultiModalDepDet)
        self.use_cross_modal_attention = use_cross_modal_attention
        if use_cross_modal_attention:
            self.cross_modal_attn = Attention(
                in_dim_k=bilstm_config['hidden_dim'] * 2,
                in_dim_q=bilstm_config['hidden_dim'] * 2,
                out_dim=bilstm_config['hidden_dim'] * 2,
                num_heads=bilstm_config['num_heads']
            )

    def forward(self, speech_features, return_all_attention=False):
        """
        Forward pass with comprehensive attention tracking

        Args:
            speech_features: MFCC features (batch, time, features)
            return_all_attention: Return all attention weights

        Returns:
            predictions: Model predictions
            attention_dict: Dictionary of attention weights for explainability
        """
        # Get predictions and attention from speech model
        logits, speech_attention = self.speech_model(
            speech_features,
            return_attention=True
        )

        attention_dict = {
            'speech_self_attention': speech_attention,
            'high_attention_regions': self.speech_model.get_high_attention_regions()
        }

        if return_all_attention:
            return logits, attention_dict
        else:
            return logits


if __name__ == '__main__':
    # Test the model
    batch_size = 8
    time_steps = 100
    n_mfcc = 40

    # Create dummy input
    x = torch.randn(batch_size, time_steps, n_mfcc)

    # Initialize model
    model = BiLSTM_MHA_Dysarthria(
        input_dim=n_mfcc,
        hidden_dim=256,
        num_layers=2,
        num_heads=8,
        num_classes=2
    )

    # Forward pass
    logits, attention_weights = model(x, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Get high attention regions
    high_attn_mask, attn_scores = model.get_high_attention_regions(threshold_percentile=75)
    print(f"High attention mask shape: {high_attn_mask.shape}")
    print(f"Attention scores shape: {attn_scores.shape}")
