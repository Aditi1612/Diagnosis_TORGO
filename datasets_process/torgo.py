"""
TORGO Dataset Loader for DeepThink-Speech
Handles MFCC extraction and preprocessing from TORGO database
"""

import os
import glob
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import re
from typing import List, Tuple, Optional, Dict
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


class TORGODataset(Dataset):
    """
    TORGO Dysarthric Speech Dataset Loader

    Directory structure expected:
    TORGO_database/
    ├── F01/  (Dysarthric Female)
    ├── F03/
    ├── F04/
    ├── FC01/ (Control Female)
    ├── FC02/
    ├── FC03/
    ├── M01/  (Dysarthric Male)
    ├── M02/
    ├── M03/
    ├── M04/
    ├── M05/
    ├── MC01/ (Control Male)
    ├── MC02/
    ├── MC03/
    └── MC04/
    """

    def __init__(self,
                 data_root: str,
                 n_mfcc: int = 40,
                 sample_rate: int = 16000,
                 max_length: int = 300,  # Max time steps
                 augment: bool = False,
                 mic_type: str = 'headMic',  # 'headMic' or 'arrayMic'
                 speaker_split: Optional[List[str]] = None):
        """
        Initialize TORGO dataset

        Args:
            data_root: Root directory of TORGO database
            n_mfcc: Number of MFCC coefficients to extract
            sample_rate: Target sample rate
            max_length: Maximum sequence length (time steps)
            augment: Apply data augmentation
            mic_type: 'headMic' or 'arrayMic'
            speaker_split: List of speaker IDs to include (for train/val/test split)
        """
        self.data_root = Path(data_root)
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.augment = augment
        self.mic_type = mic_type

        # Speaker categorization
        self.dysarthric_speakers = ['F01', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'M05']
        self.control_speakers = ['FC01', 'FC02', 'FC03', 'MC01', 'MC02', 'MC03', 'MC04']

        # Load dataset
        self.data_list = self._load_dataset(speaker_split)

        print(f"Loaded {len(self.data_list)} samples from TORGO dataset")
        print(f"  Dysarthric: {sum(1 for x in self.data_list if x['label'] == 1)}")
        print(f"  Control: {sum(1 for x in self.data_list if x['label'] == 0)}")

    def _load_dataset(self, speaker_split: Optional[List[str]]) -> List[Dict]:
        """Load all audio files and labels"""
        data_list = []
        all_wav_files = []

        # Determine which speakers to use
        if speaker_split is None:
            speakers = self.dysarthric_speakers + self.control_speakers
        else:
            speakers = speaker_split

        # First, collect all wav file paths
        for speaker_id in speakers:
            speaker_path = self.data_root / speaker_id

            if not speaker_path.exists():
                print(f"Warning: Speaker directory {speaker_id} not found")
                continue

            # Determine label
            is_dysarthric = speaker_id in self.dysarthric_speakers
            label = 1 if is_dysarthric else 0

            # Find all session directories
            session_dirs = sorted(speaker_path.glob('Session*'))

            for session_dir in session_dirs:
                # Look for wav files in appropriate microphone directory
                wav_dir = session_dir / f'wav_{self.mic_type}'

                if not wav_dir.exists():
                    continue

                # Find all .wav files
                wav_files = list(wav_dir.glob('*.wav')) + list(wav_dir.glob('*.WAV'))

                for wav_file in wav_files:
                    all_wav_files.append({
                        'path': wav_file,
                        'label': label,
                        'speaker_id': speaker_id,
                        'session': session_dir.name,
                        'is_dysarthric': is_dysarthric
                    })

        # Now validate all files with progress bar
        print(f"Validating {len(all_wav_files)} audio files...")
        for file_info in tqdm(all_wav_files, desc="Validating audio files"):
            if self._validate_audio_file(file_info['path']):
                data_list.append({
                    'audio_path': str(file_info['path']),
                    'label': file_info['label'],
                    'speaker_id': file_info['speaker_id'],
                    'session': file_info['session'],
                    'is_dysarthric': file_info['is_dysarthric']
                })

        return data_list

    def _validate_audio_file(self, audio_path: Path) -> bool:
        """
        Validate that an audio file can be loaded

        Args:
            audio_path: Path to audio file

        Returns:
            True if file can be loaded, False otherwise
        """
        try:
            # Try to load just a small portion to check if it's valid
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=0.1)
            # Check if audio is not empty
            if len(audio) == 0:
                print(f"Warning: Empty audio file: {audio_path}")
                return False
            return True
        except Exception as e:
            print(f"Warning: Could not load audio file {audio_path}: {e}")
            return False

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Get a single sample

        Returns:
            mfcc: MFCC features (max_length, n_mfcc)
            label: 0 for control, 1 for dysarthric
            metadata: Dictionary with additional info
        """
        item = self.data_list[idx]

        # Load audio
        audio, sr = librosa.load(item['audio_path'], sr=self.sample_rate)

        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment_audio(audio)

        # Extract MFCC features
        mfcc = self._extract_mfcc(audio)

        # Pad or truncate to max_length
        mfcc = self._pad_or_truncate(mfcc)

        # Convert to tensor
        mfcc_tensor = torch.FloatTensor(mfcc)
        label_tensor = torch.LongTensor([item['label']])[0]

        # Metadata
        metadata = {
            'speaker_id': item['speaker_id'],
            'session': item['session'],
            'audio_path': item['audio_path'],
            'is_dysarthric': item['is_dysarthric']
        }

        return mfcc_tensor, label_tensor, metadata

    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features

        Args:
            audio: Audio signal

        Returns:
            mfcc: MFCC features (time_steps, n_mfcc)
        """
        # Compute MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=512,
            hop_length=160,  # 10ms hop
            win_length=400   # 25ms window
        )

        # Check if MFCC has enough frames for delta computation (needs at least 9 frames)
        if mfcc.shape[1] < 9:
            # Pad MFCC to have at least 9 frames
            pad_width = 9 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='edge')

        # Add delta and delta-delta features for richer representation
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack features
        mfcc_combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

        # Transpose to (time, features)
        mfcc_combined = mfcc_combined.T

        # Normalize
        mfcc_combined = (mfcc_combined - np.mean(mfcc_combined, axis=0)) / (np.std(mfcc_combined, axis=0) + 1e-8)

        return mfcc_combined

    def _pad_or_truncate(self, mfcc: np.ndarray) -> np.ndarray:
        """Pad or truncate MFCC to fixed length"""
        current_length = mfcc.shape[0]

        if current_length > self.max_length:
            # Truncate
            mfcc = mfcc[:self.max_length, :]
        elif current_length < self.max_length:
            # Pad with zeros
            pad_length = self.max_length - current_length
            mfcc = np.pad(mfcc, ((0, pad_length), (0, 0)), mode='constant')

        return mfcc

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply MILD data augmentation to preserve discriminative features"""
        # Apply augmentations with LOWER probability to preserve class-specific features

        # 1. Time stretch (25% chance) - reduced from 80%
        if np.random.random() < 0.25:
            rate = np.random.uniform(0.9, 1.1)  # Less extreme range
            audio = librosa.effects.time_stretch(audio, rate=rate)

        # 2. Pitch shift (20% chance) - reduced from 70%
        if np.random.random() < 0.2:
            n_steps = np.random.randint(-2, 3)  # Smaller range
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)

        # 3. Add noise (30% chance) - reduced from 90%
        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.001, 0.005)  # Less noise
            noise = np.random.randn(len(audio)) * noise_level
            audio = audio + noise

        # 4. Volume scaling (25% chance) - reduced from 60%
        if np.random.random() < 0.25:
            volume_factor = np.random.uniform(0.9, 1.1)  # Less extreme
            audio = audio * volume_factor

        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def get_speaker_split(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
        """
        Create train/val/test splits based on speakers (speaker-independent)

        Args:
            train_ratio: Proportion of speakers for training
            val_ratio: Proportion of speakers for validation
            test_ratio: Proportion of speakers for testing
            seed: Random seed

        Returns:
            train_speakers, val_speakers, test_speakers
        """
        np.random.seed(seed)

        # Split dysarthric speakers
        dysarthric = self.dysarthric_speakers.copy()
        np.random.shuffle(dysarthric)
        n_dys = len(dysarthric)
        dys_train = dysarthric[:int(n_dys * train_ratio)]
        dys_val = dysarthric[int(n_dys * train_ratio):int(n_dys * (train_ratio + val_ratio))]
        dys_test = dysarthric[int(n_dys * (train_ratio + val_ratio)):]

        # Split control speakers
        control = self.control_speakers.copy()
        np.random.shuffle(control)
        n_ctrl = len(control)
        ctrl_train = control[:int(n_ctrl * train_ratio)]
        ctrl_val = control[int(n_ctrl * train_ratio):int(n_ctrl * (train_ratio + val_ratio))]
        ctrl_test = control[int(n_ctrl * (train_ratio + val_ratio)):]

        train_speakers = dys_train + ctrl_train
        val_speakers = dys_val + ctrl_val
        test_speakers = dys_test + ctrl_test

        return train_speakers, val_speakers, test_speakers


def create_torgo_dataloaders(
    data_root: str,
    batch_size: int = 16,
    n_mfcc: int = 40,
    num_workers: int = 4,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for TORGO

    Args:
        data_root: Root directory of TORGO database
        batch_size: Batch size
        n_mfcc: Number of MFCC coefficients
        num_workers: Number of data loading workers
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create temporary dataset to get speaker splits
    temp_dataset = TORGODataset(data_root, n_mfcc=n_mfcc)
    train_speakers, val_speakers, test_speakers = temp_dataset.get_speaker_split(
        train_ratio, val_ratio, test_ratio, seed
    )

    print(f"Train speakers: {train_speakers}")
    print(f"Val speakers: {val_speakers}")
    print(f"Test speakers: {test_speakers}")

    # Create datasets for each split
    train_dataset = TORGODataset(
        data_root,
        n_mfcc=n_mfcc,
        augment=True,
        speaker_split=train_speakers
    )

    val_dataset = TORGODataset(
        data_root,
        n_mfcc=n_mfcc,
        augment=False,
        speaker_split=val_speakers
    )

    test_dataset = TORGODataset(
        data_root,
        n_mfcc=n_mfcc,
        augment=False,
        speaker_split=test_speakers
    )

    # NO balanced sampling - use regular shuffle
    # Class balancing is handled ONLY by loss function weights
    # Double balancing (sampling + loss weights) causes instability
    print("\nUsing regular shuffle (NO balanced sampling)")
    print("Class imbalance handled by loss function weights only")

    train_labels = []
    for _, label, _ in train_dataset:
        train_labels.append(label.item())
    train_labels = np.array(train_labels)
    class_counts = np.bincount(train_labels)

    print(f"Training class distribution:")
    print(f"  Class 0 (Control): {class_counts[0]} samples")
    print(f"  Class 1 (Dysarthric): {class_counts[1]} samples")

    # Create dataloaders with regular shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Regular shuffle, no oversampling
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataset loader
    data_root = '/home/work/Aditi/diag_paper/TORGO_database'

    print("Creating TORGO dataloaders...")
    train_loader, val_loader, test_loader = create_torgo_dataloaders(
        data_root=data_root,
        batch_size=8,
        n_mfcc=40,
        num_workers=0
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test loading a batch
    print("\nLoading a sample batch...")
    for mfcc, labels, metadata in train_loader:
        print(f"MFCC shape: {mfcc.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        print(f"Sample metadata: {metadata['speaker_id']}")
        break
