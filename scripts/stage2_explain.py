"""
Stage 2: Clinical Explanation Generation for Wav2Vec2 Dysarthria Detection
Extracts attention from Wav2Vec2 transformer and generates LLM-based clinical explanations

Pipeline:
1. Load trained Wav2Vec2 model
2. Extract attention weights from transformer layers
3. Identify high-attention temporal regions
4. Extract acoustic features from those regions
5. Generate clinical explanations using DeepSeek-R1
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import importlib.util

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import Wav2Vec2 model
wav2vec_path = Path(__file__).parent.parent / "models" / "Wav2Vec2Classifier.py"
spec = importlib.util.spec_from_file_location("Wav2Vec2Classifier", wav2vec_path)
wav2vec_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wav2vec_module)
Wav2Vec2DysarthriaClassifier = wav2vec_module.Wav2Vec2DysarthriaClassifier

from utils.acoustic_features import AcousticFeatureExtractor
from utils.llm_explainer import ClinicalExplainer


class Wav2Vec2ExplainableInference:
    """
    Stage 2: Explainable inference pipeline for Wav2Vec2 dysarthria detection
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "facebook/wav2vec2-base",
        llm_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: str = "cuda",
        use_mfa: bool = False
    ):
        """
        Initialize the explainable inference pipeline

        Args:
            checkpoint_path: Path to trained Wav2Vec2 checkpoint
            model_name: Wav2Vec2 model name
            llm_model: LLM model for explanation generation
            device: Device for inference
            use_mfa: Whether to use Montreal Forced Aligner for phoneme analysis
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000

        print("\n" + "=" * 60)
        print("STAGE 2: CLINICAL EXPLANATION PIPELINE")
        print("=" * 60)

        # Load Wav2Vec2 model
        print("\n1. Loading trained Wav2Vec2 model...")
        self.model = self._load_model(checkpoint_path, model_name)

        # Initialize acoustic feature extractor
        print("\n2. Initializing acoustic feature extractor...")
        self.acoustic_extractor = AcousticFeatureExtractor(
            sample_rate=self.sample_rate,
            use_mfa=use_mfa
        )

        # Initialize LLM explainer
        print("\n3. Initializing clinical explainer...")
        self.clinical_explainer = ClinicalExplainer(
            model_name=llm_model,
            device=device
        )

        print("\n" + "=" * 60)
        print("Pipeline ready!")
        print("=" * 60 + "\n")

    def _load_model(self, checkpoint_path: str, model_name: str):
        """Load the trained Wav2Vec2 model with attention output support"""
        from transformers import Wav2Vec2Model, Wav2Vec2Config

        # Load config with eager attention implementation (required for attention output)
        print(f"Loading Wav2Vec2 config with eager attention...")
        config = Wav2Vec2Config.from_pretrained(model_name)
        config.output_attentions = True
        config._attn_implementation = "eager"  # Required for attention output

        # Load Wav2Vec2 with modified config
        wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, config=config)

        # Build classifier manually with the modified wav2vec2
        model = Wav2Vec2DysarthriaClassifier(
            model_name=model_name,
            num_classes=2,
            dropout=0.3,
            freeze_encoder=False,
            freeze_feature_extractor=True,
            pooling="mean"
        )

        # Replace wav2vec2 with our version that supports attention output
        model.wav2vec2 = wav2vec2

        # Load checkpoint weights
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(self.device)
        model.eval()

        print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'N/A')})")
        print(f"Attention implementation: {model.wav2vec2.config._attn_implementation}")
        return model

    def extract_attention_weights(self, audio: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract attention weights from Wav2Vec2 transformer layers

        Args:
            audio: Audio tensor (1, seq_len)

        Returns:
            logits: Model prediction logits
            attention_weights: Aggregated attention weights (seq_len,)
        """
        with torch.no_grad():
            # Get Wav2Vec2 outputs with attention
            outputs = self.model.wav2vec2(
                input_values=audio,
                output_attentions=True,
                output_hidden_states=True
            )

            hidden_states = outputs.last_hidden_state
            attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq) per layer

            # Check if attentions are valid
            if attentions is None or len(attentions) == 0:
                raise ValueError("No attention weights returned from model")

            # Filter out None values in attentions
            valid_attentions = [a for a in attentions if a is not None]
            if len(valid_attentions) == 0:
                raise ValueError("All attention layers returned None")

            # Pool hidden states for classification
            pooled = hidden_states.mean(dim=1)
            logits = self.model.classifier(pooled)

            # Aggregate attention weights across layers and heads
            # Take attention from last 4 layers (most semantic)
            num_layers = len(valid_attentions)
            last_layers = valid_attentions[-4:] if num_layers >= 4 else valid_attentions

            # Stack and average across layers and heads
            # Each attention: (batch, heads, seq, seq)
            stacked_attn = torch.stack(last_layers, dim=0)  # (layers, batch, heads, seq, seq)
            avg_attn = stacked_attn.mean(dim=(0, 2))  # Average over layers and heads -> (batch, seq, seq)

            # Get attention scores per time step (row-wise average = how much each position attends)
            attention_scores = avg_attn[0].mean(dim=0).cpu().numpy()  # (seq_len,)

            return logits, attention_scores

    def identify_high_attention_regions(
        self,
        attention_scores: np.ndarray,
        threshold_percentile: float = 75
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Identify temporal regions with high attention

        Args:
            attention_scores: Attention scores per time step
            threshold_percentile: Percentile threshold for high attention

        Returns:
            high_attention_mask: Boolean mask
            regions: List of (start, end) frame indices
        """
        threshold = np.percentile(attention_scores, threshold_percentile)
        high_attention_mask = attention_scores >= threshold

        # Find contiguous regions
        regions = []
        in_region = False
        start = 0

        for i, is_high in enumerate(high_attention_mask):
            if is_high and not in_region:
                start = i
                in_region = True
            elif not is_high and in_region:
                regions.append((start, i))
                in_region = False

        if in_region:
            regions.append((start, len(high_attention_mask)))

        return high_attention_mask, regions

    def frames_to_samples(self, frame_idx: int, audio_length: int, num_frames: int) -> int:
        """Convert frame index to sample index"""
        return int(frame_idx * audio_length / num_frames)

    def explain_single_audio(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        explanation_level: str = "clinical",
        max_length_sec: float = 4.0,
        min_length_sec: float = 0.5
    ) -> Dict:
        """
        Generate clinical explanation for a single audio file

        Args:
            audio_path: Path to audio file
            transcript: Optional transcript for phoneme alignment
            explanation_level: "clinical" or "patient"
            max_length_sec: Maximum audio length in seconds
            min_length_sec: Minimum audio length in seconds

        Returns:
            Dictionary containing prediction and explanation
        """
        print(f"\nProcessing: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Wav2Vec2 needs at least ~0.5s for CNN + transformer to work properly
        # The CNN downsamples by factor of 320, so need enough frames for attention
        min_samples = max(int(min_length_sec * self.sample_rate), 16000)  # At least 1 second
        if len(audio) < min_samples:
            # Pad short audio with zeros
            pad_length = min_samples - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            print(f"  Warning: Audio padded from {(len(audio)-pad_length)/self.sample_rate:.2f}s to {len(audio)/self.sample_rate:.2f}s")

        max_samples = int(max_length_sec * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

        # Get prediction and attention weights
        logits, attention_scores = self.extract_attention_weights(audio_tensor)

        # Get prediction
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probs[0, prediction].item()

        print(f"  Prediction: {'Dysarthric' if prediction == 1 else 'Healthy'}")
        print(f"  Confidence: {confidence:.2%}")

        # Identify high attention regions
        high_attention_mask, regions = self.identify_high_attention_regions(attention_scores)
        print(f"  High-attention regions: {len(regions)}")

        # Extract acoustic features from high-attention regions
        print("  Extracting acoustic features...")
        acoustic_features = self.acoustic_extractor.extract_all_features(
            audio,
            high_attention_mask=high_attention_mask,
            transcript=transcript,
            audio_path=audio_path,
            attention_weights=attention_scores
        )

        # Generate clinical explanation
        print("  Generating clinical explanation...")
        explanation = self.clinical_explainer.generate_explanation(
            prediction=prediction,
            confidence=confidence,
            attention_weights=attention_scores,
            acoustic_features=acoustic_features,
            audio_metadata={'audio_path': audio_path, 'transcript': transcript},
            explanation_level=explanation_level
        )

        # Add attention analysis
        explanation['attention_scores'] = attention_scores.tolist()
        explanation['high_attention_regions'] = regions
        explanation['num_high_attention_frames'] = int(high_attention_mask.sum())

        return explanation

    def explain_batch(
        self,
        audio_paths: List[str],
        transcripts: Optional[List[str]] = None,
        explanation_level: str = "clinical",
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate explanations for multiple audio files

        Args:
            audio_paths: List of audio file paths
            transcripts: Optional list of transcripts
            explanation_level: "clinical" or "patient"
            output_file: Optional path to save results

        Returns:
            List of explanation dictionaries
        """
        results = []

        for i, audio_path in enumerate(tqdm(audio_paths, desc="Generating explanations")):
            transcript = transcripts[i] if transcripts else None

            try:
                explanation = self.explain_single_audio(
                    audio_path,
                    transcript=transcript,
                    explanation_level=explanation_level
                )
                results.append(explanation)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'audio_path': audio_path,
                    'error': str(e)
                })

        # Save results if output file specified
        if output_file:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = []
            for r in results:
                sr = {}
                for k, v in r.items():
                    if isinstance(v, np.ndarray):
                        sr[k] = v.tolist()
                    elif isinstance(v, dict):
                        sr[k] = {
                            kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                            for kk, vv in v.items()
                        }
                    else:
                        sr[k] = v
                serializable_results.append(sr)

            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")

        return results

    def load_llm(self):
        """Load the LLM model for explanation generation"""
        print("\nLoading LLM for explanation generation...")
        self.clinical_explainer.load_model()
        print("LLM loaded successfully!")


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Clinical Explanation Generation')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained Wav2Vec2 checkpoint')
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base',
                        help='Wav2Vec2 model name')

    # LLM arguments
    parser.add_argument('--llm_model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
                        help='LLM model for explanation generation')
    parser.add_argument('--load_llm', action='store_true',
                        help='Load LLM model (requires significant GPU memory)')

    # Input arguments
    parser.add_argument('--audio', type=str, default=None,
                        help='Path to single audio file')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Directory containing audio files')
    parser.add_argument('--transcript', type=str, default=None,
                        help='Transcript for single audio file')

    # Output arguments
    parser.add_argument('--output', type=str, default='./output/explanations.json',
                        help='Output file for explanations')
    parser.add_argument('--explanation_level', type=str, default='clinical',
                        choices=['clinical', 'patient'],
                        help='Explanation level (clinical for SLPs, patient for patients)')

    # Other arguments
    parser.add_argument('--use_mfa', action='store_true',
                        help='Use Montreal Forced Aligner for phoneme analysis')
    parser.add_argument('--max_length_sec', type=float, default=4.0,
                        help='Maximum audio length in seconds')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = Wav2Vec2ExplainableInference(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        llm_model=args.llm_model,
        use_mfa=args.use_mfa
    )

    # Load LLM if requested
    if args.load_llm:
        pipeline.load_llm()

    # Process audio
    if args.audio:
        # Single audio file
        explanation = pipeline.explain_single_audio(
            args.audio,
            transcript=args.transcript,
            explanation_level=args.explanation_level,
            max_length_sec=args.max_length_sec
        )

        # Print explanation
        print("\n" + "=" * 60)
        print("CLINICAL EXPLANATION")
        print("=" * 60)
        print(explanation['explanation'])
        print("=" * 60)

        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(explanation, f, indent=2, default=str)
        print(f"\nFull explanation saved to: {args.output}")

    elif args.audio_dir:
        # Directory of audio files
        audio_dir = Path(args.audio_dir)
        audio_paths = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.WAV'))

        if not audio_paths:
            print(f"No audio files found in {args.audio_dir}")
            return

        print(f"Found {len(audio_paths)} audio files")

        results = pipeline.explain_batch(
            [str(p) for p in audio_paths],
            explanation_level=args.explanation_level,
            output_file=args.output
        )

        print(f"\nProcessed {len(results)} files")

    else:
        print("Please specify --audio or --audio_dir")


if __name__ == '__main__':
    main()
