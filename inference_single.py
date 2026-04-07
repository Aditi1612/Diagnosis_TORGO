#!/usr/bin/env python
"""
Single File Inference for DeepThink-Speech
Usage: python inference_single.py --checkpoint path/to/model.pth --audio path/to/audio.wav
"""

import argparse
import torch
import librosa
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import model
import importlib.util
deepthink_path = Path(__file__).parent / "models" / "DeepThinkSpeech.py"
spec = importlib.util.spec_from_file_location("DeepThinkSpeech", deepthink_path)
deepthink_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepthink_module)
BiLSTM_MHA_Dysarthria = deepthink_module.BiLSTM_MHA_Dysarthria


def extract_mfcc_features(audio_path, n_mfcc=40, sample_rate=16000, max_length=300):
    """Extract MFCC features from audio file"""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate)

    # Compute MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=512,
        hop_length=160,
        win_length=400
    )

    # Check if MFCC has enough frames for delta computation
    if mfcc.shape[1] < 9:
        pad_width = 9 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='edge')

    # Add delta and delta-delta features
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack features
    mfcc_combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

    # Transpose to (time, features)
    mfcc_combined = mfcc_combined.T

    # Normalize
    mfcc_combined = (mfcc_combined - np.mean(mfcc_combined, axis=0)) / (np.std(mfcc_combined, axis=0) + 1e-8)

    # Pad or truncate to max_length
    current_length = mfcc_combined.shape[0]
    if current_length > max_length:
        mfcc_combined = mfcc_combined[:max_length, :]
    elif current_length < max_length:
        pad_length = max_length - current_length
        mfcc_combined = np.pad(mfcc_combined, ((0, pad_length), (0, 0)), mode='constant')

    return mfcc_combined


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model args from checkpoint
    args = checkpoint.get('args', {})
    n_mfcc = args.get('n_mfcc', 40)
    hidden_dim = args.get('hidden_dim', 128)
    num_layers = args.get('num_layers', 2)
    num_heads = args.get('num_heads', 4)
    dropout = args.get('dropout', 0.5)

    # Initialize model
    model = BiLSTM_MHA_Dysarthria(
        input_dim=n_mfcc * 3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_classes=2
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, args


def predict(model, mfcc_features, device='cuda'):
    """Run inference on MFCC features"""
    # Convert to tensor
    mfcc_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits, attention_weights = model(mfcc_tensor, return_attention=True)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probs[0, prediction].item()

    return prediction, confidence, attention_weights.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Single audio file inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file (.wav)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Check files exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("DeepThink-Speech: Single File Inference")
    print("="*60)
    print(f"Audio file: {args.audio}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading model...")
    model, model_args = load_model(args.checkpoint, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # Extract features
    print("Extracting MFCC features...")
    n_mfcc = model_args.get('n_mfcc', 40)
    mfcc_features = extract_mfcc_features(args.audio, n_mfcc=n_mfcc)
    print(f"Features shape: {mfcc_features.shape}")
    print()

    # Predict
    print("Running inference...")
    prediction, confidence, attention_weights = predict(model, mfcc_features, device)

    # Results
    print("="*60)
    print("RESULTS")
    print("="*60)
    label = "Dysarthric" if prediction == 1 else "Healthy (Control)"
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print()

    # Attention analysis
    attention_mean = attention_weights.mean(axis=1).flatten()
    high_attention_frames = np.where(attention_mean > attention_mean.mean())[0]
    print(f"High attention frames: {len(high_attention_frames)}/{len(attention_mean)}")
    print(f"Attention focus: frames {high_attention_frames[:10].tolist()[:5]}...")
    print("="*60)


if __name__ == '__main__':
    main()
