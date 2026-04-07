#!/usr/bin/env python3
"""
Run Stage 2 Clinical Explanations on TORGO Test Set
Generates LLM-based clinical explanations for the best performing model
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add parent directory
sys.path.append(str(Path(__file__).parent))

from scripts.stage2_explain import Wav2Vec2ExplainableInference


def load_torgo_test_set(data_root: str, split_mode: str = "speaker"):
    """Load TORGO test set file paths"""
    from datasets_process.torgo_raw import create_torgo_raw_dataloaders
    
    # Get dataloaders
    _, _, test_loader = create_torgo_raw_dataloaders(
        data_root=data_root,
        batch_size=1,
        num_workers=0,
        split_mode=split_mode,
        max_length_sec=4.0,
        seed=42
    )
    
    # Extract audio paths and labels from dataset
    test_dataset = test_loader.dataset
    audio_paths = [sample['audio_path'] for sample in test_dataset.data_list]
    labels = [sample['label'] for sample in test_dataset.data_list]
    speaker_ids = list(set([sample['speaker_id'] for sample in test_dataset.data_list]))
    
    print(f"\nLoaded TORGO test set:")
    print(f"  Total samples: {len(audio_paths)}")
    print(f"  Dysarthric: {sum(labels)}")
    print(f"  Control: {len(labels) - sum(labels)}")
    print(f"  Speakers: {sorted(speaker_ids)}")
    
    return audio_paths, labels


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Clinical Explanations on TORGO Test Set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained Wav2Vec2 checkpoint')
    parser.add_argument('--data_root', type=str, default='./TORGO_database',
                        help='Path to TORGO database')
    parser.add_argument('--output_dir', type=str, default='./output/stage2_explanations',
                        help='Output directory for explanations')
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base',
                        help='Wav2Vec2 model name')
    parser.add_argument('--llm_model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
                        help='LLM model for explanation generation')
    parser.add_argument('--split_mode', type=str, default='speaker',
                        help='Split mode (speaker or sample)')
    parser.add_argument('--explanation_level', type=str, default='clinical',
                        choices=['clinical', 'patient'],
                        help='Explanation level')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (None = all)')
    parser.add_argument('--use_llm', action='store_true',
                        help='Enable LLM explanation generation (requires significant GPU memory)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("STAGE 2: CLINICAL EXPLANATION GENERATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"LLM Enabled: {args.use_llm}")
    print("=" * 70 + "\n")
    
    # Initialize pipeline
    pipeline = Wav2Vec2ExplainableInference(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        llm_model=args.llm_model,
        use_mfa=False
    )
    
    # Load LLM if requested
    if args.use_llm:
        pipeline.load_llm()
    
    # Load test set
    print("\nLoading TORGO test set...")
    audio_paths, labels = load_torgo_test_set(args.data_root, args.split_mode)
    
    # Limit samples if specified
    if args.num_samples:
        audio_paths = audio_paths[:args.num_samples]
        labels = labels[:args.num_samples]
        print(f"Processing first {args.num_samples} samples")
    
    # Process each audio file
    results = []
    correct = 0
    total = 0
    
    print(f"\nProcessing {len(audio_paths)} audio files...")
    for audio_path, true_label in tqdm(zip(audio_paths, labels), total=len(audio_paths)):
        try:
            explanation = pipeline.explain_single_audio(
                audio_path=str(audio_path),
                explanation_level=args.explanation_level,
                max_length_sec=4.0
            )
            
            # Add ground truth
            explanation['ground_truth'] = 'Dysarthric' if true_label == 1 else 'Control'
            explanation['audio_path'] = str(audio_path)
            
            # Check accuracy
            pred_label = 1 if explanation['prediction'] == 'Dysarthric' else 0
            if pred_label == true_label:
                correct += 1
            total += 1
            
            results.append(explanation)
            
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            continue
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Save results
    output_file = output_dir / 'test_explanations.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary
    summary = {
        'total_samples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'checkpoint': args.checkpoint,
        'model_name': args.model_name,
        'llm_enabled': args.use_llm
    }
    
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("STAGE 2 COMPLETE")
    print("=" * 70)
    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
