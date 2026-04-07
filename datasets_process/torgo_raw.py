"""
TORGO Dataset Loader for Wav2Vec2
Loads raw audio waveforms (no MFCC) for pretrained speech models
"""

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


class TORGORawDataset(Dataset):
    """
    TORGO Dataset for Wav2Vec2/HuBERT models
    Returns raw audio waveforms instead of MFCC features
    """

    def __init__(
        self,
        data_root: str,
        sample_rate: int = 16000,
        max_length_sec: float = 5.0,  # Max audio length in seconds
        augment: bool = False,
        mic_type: str = 'headMic',
        speaker_split: Optional[List[str]] = None
    ):
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.max_length = int(max_length_sec * sample_rate)
        self.augment = augment
        self.mic_type = mic_type

        # Speaker categorization
        self.dysarthric_speakers = ['F01', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'M05']
        self.control_speakers = ['FC01', 'FC02', 'FC03', 'MC01', 'MC02', 'MC03', 'MC04']

        # Load dataset
        self.data_list = self._load_dataset(speaker_split)

        print(f"Loaded {len(self.data_list)} samples (raw audio)")
        print(f"  Dysarthric: {sum(1 for x in self.data_list if x['label'] == 1)}")
        print(f"  Control: {sum(1 for x in self.data_list if x['label'] == 0)}")

    def _load_dataset(self, speaker_split: Optional[List[str]]) -> List[Dict]:
        """Load all audio file paths and labels"""
        data_list = []

        if speaker_split is None:
            speakers = self.dysarthric_speakers + self.control_speakers
        else:
            speakers = speaker_split

        all_files = []
        for speaker_id in speakers:
            speaker_path = self.data_root / speaker_id

            if not speaker_path.exists():
                print(f"Warning: Speaker directory {speaker_id} not found")
                continue

            is_dysarthric = speaker_id in self.dysarthric_speakers
            label = 1 if is_dysarthric else 0

            session_dirs = sorted(speaker_path.glob('Session*'))

            for session_dir in session_dirs:
                wav_dir = session_dir / f'wav_{self.mic_type}'

                if not wav_dir.exists():
                    continue

                wav_files = list(wav_dir.glob('*.wav')) + list(wav_dir.glob('*.WAV'))

                for wav_file in wav_files:
                    all_files.append({
                        'path': wav_file,
                        'label': label,
                        'speaker_id': speaker_id,
                        'session': session_dir.name,
                        'is_dysarthric': is_dysarthric
                    })

        # Validate files
        print(f"Validating {len(all_files)} audio files...")
        for file_info in tqdm(all_files, desc="Validating"):
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
        """Check if audio file can be loaded"""
        try:
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=0.1)
            return len(audio) > 0
        except Exception:
            return False

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Get a single sample

        Returns:
            audio: Raw audio waveform (max_length,)
            label: 0 for control, 1 for dysarthric
            metadata: Dictionary with additional info
        """
        item = self.data_list[idx]

        # Load audio
        audio, sr = librosa.load(item['audio_path'], sr=self.sample_rate)

        # Simple augmentation for raw audio
        if self.augment:
            audio = self._augment_audio(audio)

        # Pad or truncate to fixed length
        audio = self._pad_or_truncate(audio)

        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio)
        label_tensor = torch.LongTensor([item['label']])[0]

        metadata = {
            'speaker_id': item['speaker_id'],
            'session': item['session'],
            'audio_path': item['audio_path'],
            'is_dysarthric': item['is_dysarthric']
        }

        return audio_tensor, label_tensor, metadata

    def _pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """Pad or truncate audio to fixed length"""
        if len(audio) > self.max_length:
            # Random crop for augmentation, center crop otherwise
            if self.augment:
                start = np.random.randint(0, len(audio) - self.max_length)
                audio = audio[start:start + self.max_length]
            else:
                # Center crop
                start = (len(audio) - self.max_length) // 2
                audio = audio[start:start + self.max_length]
        elif len(audio) < self.max_length:
            # Pad with zeros
            pad_length = self.max_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')

        return audio

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Light augmentation for raw audio"""
        # Only light augmentation - Wav2Vec2 is sensitive to heavy changes

        # Small noise (20% chance)
        if np.random.random() < 0.2:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise

        # Small volume change (20% chance)
        if np.random.random() < 0.2:
            volume = np.random.uniform(0.9, 1.1)
            audio = audio * volume

        return np.clip(audio, -1.0, 1.0)

    def get_speaker_split(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
        """
        Create speaker-independent train/val/test splits

        TORGO speaker severity levels (for balanced splits):
        - Severe: F01, M01, M02, M04
        - Moderate: F03, M03, M05
        - Mild: F04
        - Control: FC01, FC02, FC03, MC01, MC02, MC03, MC04

        We ensure each split has a mix of severity levels for fair evaluation.
        """
        np.random.seed(seed)

        # Group dysarthric speakers by severity for stratified split
        severe = ['F01', 'M01', 'M02', 'M04']      # 4 speakers
        moderate = ['F03', 'M03', 'M05']           # 3 speakers
        mild = ['F04']                              # 1 speaker
        control = self.control_speakers.copy()     # 7 speakers

        np.random.shuffle(severe)
        np.random.shuffle(moderate)
        np.random.shuffle(control)

        # Balanced split ensuring each set has mix of severities:
        # Train: 3 severe + 2 moderate + 1 mild + 4 control = 10 speakers
        # Val:   1 severe + 1 moderate + 0 mild + 2 control = 4 speakers
        # Test:  0 severe + 0 moderate + 0 mild + 1 control = 1 speaker (too few!)

        # Better split for small dataset (15 speakers total):
        # Train: 5 dysarthric + 4 control = 9 speakers (~60%)
        # Val:   2 dysarthric + 2 control = 4 speakers (~27%)
        # Test:  1 dysarthric + 1 control = 2 speakers (~13%)

        # Stratified by severity:
        train_speakers = severe[:2] + moderate[:2] + mild[:1] + control[:4]  # 9 speakers
        val_speakers = severe[2:3] + moderate[2:3] + control[4:6]            # 4 speakers
        test_speakers = severe[3:4] + control[6:7]                           # 2 speakers

        print(f"\nSpeaker split (severity-balanced):")
        print(f"  Train ({len(train_speakers)}): {train_speakers}")
        print(f"  Val ({len(val_speakers)}): {val_speakers}")
        print(f"  Test ({len(test_speakers)}): {test_speakers}")

        return train_speakers, val_speakers, test_speakers


def create_torgo_raw_dataloaders(
    data_root: str,
    batch_size: int = 8,
    max_length_sec: float = 5.0,
    num_workers: int = 4,
    seed: int = 42,
    split_mode: str = 'speaker'  # 'speaker' (fair evaluation) or 'sample' (quick dev only)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for TORGO (raw audio)

    Args:
        split_mode: 'sample' - mix speakers across splits for better generalization
                    'speaker' - speaker-independent splits (harder, for final evaluation)
    """

    if split_mode == 'sample':
        # SAMPLE-LEVEL SPLIT: Mix all speakers across train/val/test
        # This helps the model learn from all speakers and generalize better
        print("\n" + "="*50)
        print("Using SAMPLE-LEVEL split (speakers mixed across splits)")
        print("="*50)

        # Load all data from all speakers
        full_dataset = TORGORawDataset(
            data_root,
            max_length_sec=max_length_sec,
            augment=False,
            speaker_split=None  # All speakers
        )

        # Create stratified split (maintain class balance in each split)
        np.random.seed(seed)

        # Separate indices by class for stratified split
        dysarthric_indices = [i for i, item in enumerate(full_dataset.data_list) if item['label'] == 1]
        control_indices = [i for i, item in enumerate(full_dataset.data_list) if item['label'] == 0]

        np.random.shuffle(dysarthric_indices)
        np.random.shuffle(control_indices)

        # Split each class 70/15/15
        def split_indices(indices, train_r=0.7, val_r=0.15):
            n = len(indices)
            train_end = int(n * train_r)
            val_end = int(n * (train_r + val_r))
            return indices[:train_end], indices[train_end:val_end], indices[val_end:]

        dys_train, dys_val, dys_test = split_indices(dysarthric_indices)
        ctrl_train, ctrl_val, ctrl_test = split_indices(control_indices)

        train_indices = dys_train + ctrl_train
        val_indices = dys_val + ctrl_val
        test_indices = dys_test + ctrl_test

        np.random.shuffle(train_indices)

        print(f"\nSample distribution:")
        print(f"  Train: {len(train_indices)} ({len(dys_train)} dys + {len(ctrl_train)} ctrl)")
        print(f"  Val:   {len(val_indices)} ({len(dys_val)} dys + {len(ctrl_val)} ctrl)")
        print(f"  Test:  {len(test_indices)} ({len(dys_test)} dys + {len(ctrl_test)} ctrl)")

        # Check speaker distribution in each split
        def get_speakers(indices):
            return set(full_dataset.data_list[i]['speaker_id'] for i in indices)

        print(f"\nSpeakers in train: {sorted(get_speakers(train_indices))}")
        print(f"Speakers in val:   {sorted(get_speakers(val_indices))}")
        print(f"Speakers in test:  {sorted(get_speakers(test_indices))}")

        # Create subset datasets
        from torch.utils.data import Subset

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # Enable augmentation for training by wrapping
        class AugmentedSubset(Dataset):
            def __init__(self, subset, base_dataset, augment=True):
                self.subset = subset
                self.base_dataset = base_dataset
                self.augment = augment

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                real_idx = self.subset.indices[idx]
                item = self.base_dataset.data_list[real_idx]

                # Load audio
                audio, sr = librosa.load(item['audio_path'], sr=self.base_dataset.sample_rate)

                # Apply augmentation if enabled
                if self.augment:
                    audio = self.base_dataset._augment_audio(audio)

                # Pad or truncate
                audio = self.base_dataset._pad_or_truncate(audio)

                audio_tensor = torch.FloatTensor(audio)
                label_tensor = torch.LongTensor([item['label']])[0]

                metadata = {
                    'speaker_id': item['speaker_id'],
                    'session': item['session'],
                    'audio_path': item['audio_path'],
                    'is_dysarthric': item['is_dysarthric']
                }

                return audio_tensor, label_tensor, metadata

        train_dataset = AugmentedSubset(train_dataset, full_dataset, augment=True)
        val_dataset = AugmentedSubset(val_dataset, full_dataset, augment=False)
        test_dataset = AugmentedSubset(test_dataset, full_dataset, augment=False)

        # Use WeightedRandomSampler for balanced training
        train_labels = [full_dataset.data_list[i]['label'] for i in train_indices]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        print(f"\nUsing WeightedRandomSampler for balanced batches")
        print(f"  Class weights: Control={class_weights[0]:.4f}, Dysarthric={class_weights[1]:.4f}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Balanced sampling
            num_workers=num_workers,
            pin_memory=True
        )

    else:
        # SPEAKER-LEVEL SPLIT: Original speaker-independent approach
        print("\n" + "="*50)
        print("Using SPEAKER-LEVEL split (speaker-independent)")
        print("="*50)

        temp_dataset = TORGORawDataset(data_root, max_length_sec=max_length_sec)
        train_speakers, val_speakers, test_speakers = temp_dataset.get_speaker_split(seed=seed)

        print(f"\nSpeaker splits:")
        print(f"  Train: {train_speakers}")
        print(f"  Val: {val_speakers}")
        print(f"  Test: {test_speakers}")

        train_dataset = TORGORawDataset(
            data_root,
            max_length_sec=max_length_sec,
            augment=True,
            speaker_split=train_speakers
        )

        val_dataset = TORGORawDataset(
            data_root,
            max_length_sec=max_length_sec,
            augment=False,
            speaker_split=val_speakers
        )

        test_dataset = TORGORawDataset(
            data_root,
            max_length_sec=max_length_sec,
            augment=False,
            speaker_split=test_speakers
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
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
    # Test the dataset
    data_root = '/home/work/Aditi/diag_paper/TORGO_database'

    print("Creating raw audio dataloaders...")
    train_loader, val_loader, test_loader = create_torgo_raw_dataloaders(
        data_root=data_root,
        batch_size=4,
        max_length_sec=3.0,
        num_workers=0
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test loading a batch
    print("\nLoading a sample batch...")
    for audio, labels, metadata in train_loader:
        print(f"Audio shape: {audio.shape}")
        print(f"Labels: {labels}")
        print(f"Speakers: {metadata['speaker_id']}")
        break
