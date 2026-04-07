"""
Montreal Forced Aligner (MFA) Integration for DeepThink-Speech
Performs forced phoneme alignment for phoneme-level dysarthria analysis

This module provides phoneme-level timing information by aligning transcripts
to audio using MFA's GMM-HMM acoustic models.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil
import pandas as pd
import numpy as np


class MontrealForcedAligner:
    """
    Interface to Montreal Forced Aligner for phoneme-level alignment

    MFA uses GMM-HMM acoustic models to align known transcripts to audio,
    providing precise phoneme-level timing information crucial for
    dysarthria analysis.
    """

    def __init__(self,
                 mfa_path: Optional[str] = None,
                 acoustic_model: str = "english_us_arpa",
                 dictionary: str = "english_us_arpa"):
        """
        Initialize MFA wrapper

        Args:
            mfa_path: Path to MFA executable (if None, assumes 'mfa' in PATH)
            acoustic_model: Pretrained acoustic model name
            dictionary: Pronunciation dictionary name
        """
        self.mfa_path = mfa_path or "mfa"
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary

        # Check if MFA is installed
        self._check_mfa_installation()

    def _check_mfa_installation(self):
        """Check if MFA is properly installed"""
        try:
            result = subprocess.run(
                [self.mfa_path, "version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"MFA version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Montreal Forced Aligner not found. Please install MFA:\n"
                f"  conda install -c conda-forge montreal-forced-aligner\n"
                f"Error: {e}"
            )

    def download_models(self):
        """Download pretrained acoustic model and dictionary if needed"""
        print(f"Downloading MFA acoustic model: {self.acoustic_model}")
        try:
            subprocess.run(
                [self.mfa_path, "model", "download", "acoustic", self.acoustic_model],
                check=True
            )
            print(f"Downloading MFA dictionary: {self.dictionary}")
            subprocess.run(
                [self.mfa_path, "model", "download", "dictionary", self.dictionary],
                check=True
            )
            print("Models downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading models: {e}")
            print("Models may already be downloaded.")

    def align_corpus(self,
                     audio_dir: str,
                     transcript_dir: str,
                     output_dir: str,
                     speaker_characters: int = 0) -> str:
        """
        Align an entire corpus using MFA

        Args:
            audio_dir: Directory containing .wav files
            transcript_dir: Directory containing .txt transcript files
            output_dir: Directory for alignment output
            speaker_characters: Number of characters for speaker identification

        Returns:
            Path to output directory with TextGrid files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Running MFA alignment...")
        print(f"  Audio: {audio_dir}")
        print(f"  Transcripts: {transcript_dir}")
        print(f"  Output: {output_dir}")

        try:
            cmd = [
                self.mfa_path, "align",
                audio_dir,
                self.dictionary,
                self.acoustic_model,
                output_dir,
                "--clean"
            ]

            if speaker_characters > 0:
                cmd.extend(["--speaker_characters", str(speaker_characters)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            print("Alignment completed successfully")
            return output_dir

        except subprocess.CalledProcessError as e:
            print(f"MFA alignment failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise

    def align_single_file(self,
                         audio_path: str,
                         transcript: str,
                         output_dir: Optional[str] = None) -> Dict:
        """
        Align a single audio file with its transcript

        Args:
            audio_path: Path to audio file
            transcript: Text transcript
            output_dir: Optional output directory (uses temp dir if None)

        Returns:
            Dictionary with phoneme-level alignment information
        """
        # Create temporary directory structure
        use_temp = output_dir is None
        if use_temp:
            temp_dir = tempfile.mkdtemp()
            work_dir = Path(temp_dir)
        else:
            work_dir = Path(output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Setup directory structure
            audio_dir = work_dir / "audio"
            transcript_dir = work_dir / "transcripts"
            output_dir_path = work_dir / "output"

            audio_dir.mkdir(exist_ok=True)
            transcript_dir.mkdir(exist_ok=True)
            output_dir_path.mkdir(exist_ok=True)

            # Copy audio file
            audio_path_obj = Path(audio_path)
            audio_filename = audio_path_obj.stem
            shutil.copy(audio_path, audio_dir / audio_path_obj.name)

            # Write transcript
            transcript_file = transcript_dir / f"{audio_filename}.txt"
            with open(transcript_file, 'w') as f:
                f.write(transcript.strip())

            # Run alignment
            self.align_corpus(
                str(audio_dir),
                str(transcript_dir),
                str(output_dir_path)
            )

            # Parse TextGrid output
            textgrid_file = output_dir_path / f"{audio_filename}.TextGrid"
            if textgrid_file.exists():
                alignment = self.parse_textgrid(str(textgrid_file))
                return alignment
            else:
                raise RuntimeError(f"TextGrid file not found: {textgrid_file}")

        finally:
            # Cleanup temp directory
            if use_temp and work_dir.exists():
                shutil.rmtree(work_dir)

    def parse_textgrid(self, textgrid_path: str) -> Dict:
        """
        Parse TextGrid file to extract phoneme alignments

        Args:
            textgrid_path: Path to TextGrid file

        Returns:
            Dictionary with word and phoneme alignments
        """
        try:
            import textgrid
        except ImportError:
            raise ImportError(
                "textgrid package required for parsing. Install with:\n"
                "  pip install praat-textgrids"
            )

        tg = textgrid.TextGrid.fromFile(textgrid_path)

        alignment = {
            'words': [],
            'phones': [],
            'duration': 0.0
        }

        # Extract word tier
        for tier in tg:
            if tier.name.lower() == 'words':
                for interval in tier:
                    if interval.mark.strip():  # Skip empty intervals
                        alignment['words'].append({
                            'word': interval.mark,
                            'start': interval.minTime,
                            'end': interval.maxTime,
                            'duration': interval.maxTime - interval.minTime
                        })
                alignment['duration'] = tier.maxTime

            # Extract phone tier
            elif tier.name.lower() == 'phones':
                for interval in tier:
                    if interval.mark.strip():  # Skip empty intervals
                        alignment['phones'].append({
                            'phone': interval.mark,
                            'start': interval.minTime,
                            'end': interval.maxTime,
                            'duration': interval.maxTime - interval.minTime
                        })

        return alignment

    def get_phoneme_at_time(self, alignment: Dict, time: float) -> Optional[str]:
        """Get phoneme at specific time point"""
        for phone in alignment['phones']:
            if phone['start'] <= time <= phone['end']:
                return phone['phone']
        return None

    def get_phonemes_in_window(self, alignment: Dict,
                               start_time: float,
                               end_time: float) -> List[Dict]:
        """Get all phonemes within a time window"""
        phonemes = []
        for phone in alignment['phones']:
            # Check if phoneme overlaps with window
            if phone['start'] <= end_time and phone['end'] >= start_time:
                phonemes.append(phone)
        return phonemes

    def map_attention_to_phonemes(self,
                                  alignment: Dict,
                                  attention_weights: np.ndarray,
                                  audio_duration: float,
                                  hop_length: int = 160,
                                  sample_rate: int = 16000) -> Dict:
        """
        Map attention weights to phonemes

        Args:
            alignment: Phoneme alignment from MFA
            attention_weights: Attention weights from model (time_steps,)
            audio_duration: Duration of audio in seconds
            hop_length: Hop length used for MFCC extraction
            sample_rate: Audio sample rate

        Returns:
            Dictionary mapping phonemes to average attention weights
        """
        # Calculate time per attention frame
        time_per_frame = hop_length / sample_rate

        # Initialize phoneme attention accumulator
        phoneme_attention = {}
        phoneme_counts = {}

        for frame_idx, attn_weight in enumerate(attention_weights):
            # Get time for this frame
            frame_time = frame_idx * time_per_frame

            # Find corresponding phoneme
            phoneme = self.get_phoneme_at_time(alignment, frame_time)

            if phoneme:
                if phoneme not in phoneme_attention:
                    phoneme_attention[phoneme] = 0.0
                    phoneme_counts[phoneme] = 0

                phoneme_attention[phoneme] += attn_weight
                phoneme_counts[phoneme] += 1

        # Calculate average attention per phoneme
        phoneme_attention_avg = {
            phone: phoneme_attention[phone] / phoneme_counts[phone]
            for phone in phoneme_attention
        }

        # Sort by attention weight
        sorted_phonemes = sorted(
            phoneme_attention_avg.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'phoneme_attention': dict(sorted_phonemes),
            'top_phonemes': [p[0] for p in sorted_phonemes[:10]],
            'phoneme_counts': phoneme_counts
        }

    def analyze_phoneme_classes(self, phoneme_attention: Dict) -> Dict:
        """
        Analyze attention by phoneme classes (vowels, consonants, etc.)

        Args:
            phoneme_attention: Output from map_attention_to_phonemes

        Returns:
            Dictionary with phoneme class analysis
        """
        # ARPAbet phoneme classes
        vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY',
                 'OW', 'OY', 'UH', 'UW'}
        stops = {'B', 'D', 'G', 'P', 'T', 'K'}
        fricatives = {'DH', 'F', 'S', 'SH', 'TH', 'V', 'Z', 'ZH', 'HH'}
        nasals = {'M', 'N', 'NG'}
        liquids = {'L', 'R'}
        glides = {'W', 'Y'}

        class_attention = {
            'vowels': [],
            'stops': [],
            'fricatives': [],
            'nasals': [],
            'liquids': [],
            'glides': [],
            'other': []
        }

        for phone, attn in phoneme_attention['phoneme_attention'].items():
            # Remove stress markers (0, 1, 2)
            phone_base = phone.rstrip('012')

            if phone_base in vowels:
                class_attention['vowels'].append(attn)
            elif phone_base in stops:
                class_attention['stops'].append(attn)
            elif phone_base in fricatives:
                class_attention['fricatives'].append(attn)
            elif phone_base in nasals:
                class_attention['nasals'].append(attn)
            elif phone_base in liquids:
                class_attention['liquids'].append(attn)
            elif phone_base in glides:
                class_attention['glides'].append(attn)
            else:
                class_attention['other'].append(attn)

        # Calculate average attention per class
        class_avg = {
            cls: np.mean(attn_list) if attn_list else 0.0
            for cls, attn_list in class_attention.items()
        }

        return {
            'class_attention': class_avg,
            'highest_attention_class': max(class_avg.items(), key=lambda x: x[1])[0],
            'class_counts': {cls: len(attn_list) for cls, attn_list in class_attention.items()}
        }


class TORGOTranscriptExtractor:
    """
    Extract transcripts from TORGO database for MFA alignment

    TORGO includes orthographic transcriptions in prompts/ directories
    """

    def __init__(self, torgo_root: str):
        self.torgo_root = Path(torgo_root)

    def extract_transcript(self, audio_path: str) -> Optional[str]:
        """
        Extract transcript for a given TORGO audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Transcript text or None if not found
        """
        audio_path = Path(audio_path)

        # Navigate to prompts directory
        # Structure: Speaker/Session/wav_headMic/file.wav
        #         -> Speaker/Session/prompts/file.txt

        session_dir = audio_path.parent.parent
        prompts_dir = session_dir / "prompts"

        # Look for corresponding transcript
        transcript_file = prompts_dir / f"{audio_path.stem}.txt"

        if transcript_file.exists():
            with open(transcript_file, 'r') as f:
                transcript = f.read().strip()
            return transcript
        else:
            return None

    def prepare_corpus_for_mfa(self,
                               speaker_ids: List[str],
                               output_audio_dir: str,
                               output_transcript_dir: str,
                               mic_type: str = "headMic") -> int:
        """
        Prepare TORGO corpus for MFA alignment

        Args:
            speaker_ids: List of speaker IDs to process
            output_audio_dir: Directory for audio files
            output_transcript_dir: Directory for transcript files
            mic_type: 'headMic' or 'arrayMic'

        Returns:
            Number of files prepared
        """
        audio_dir = Path(output_audio_dir)
        transcript_dir = Path(output_transcript_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)

        file_count = 0

        for speaker_id in speaker_ids:
            speaker_path = self.torgo_root / speaker_id

            if not speaker_path.exists():
                print(f"Warning: Speaker {speaker_id} not found")
                continue

            # Find all sessions
            session_dirs = sorted(speaker_path.glob("Session*"))

            for session_dir in session_dirs:
                wav_dir = session_dir / f"wav_{mic_type}"
                prompts_dir = session_dir / "prompts"

                if not wav_dir.exists() or not prompts_dir.exists():
                    continue

                # Process all audio files
                for wav_file in wav_dir.glob("*.wav"):
                    # Find transcript
                    transcript_file = prompts_dir / f"{wav_file.stem}.txt"

                    if transcript_file.exists():
                        # Copy audio
                        output_audio = audio_dir / f"{speaker_id}_{session_dir.name}_{wav_file.name}"
                        shutil.copy(wav_file, output_audio)

                        # Copy/clean transcript
                        with open(transcript_file, 'r') as f:
                            transcript = f.read().strip()

                        output_transcript = transcript_dir / f"{speaker_id}_{session_dir.name}_{wav_file.stem}.txt"
                        with open(output_transcript, 'w') as f:
                            f.write(transcript)

                        file_count += 1

        print(f"Prepared {file_count} files for MFA alignment")
        return file_count


if __name__ == '__main__':
    # Example usage
    print("Montreal Forced Aligner Integration for DeepThink-Speech")
    print("=" * 60)

    # Initialize MFA
    mfa = MontrealForcedAligner()

    # Download models (only needed once)
    # mfa.download_models()

    # Example: Prepare TORGO corpus for alignment
    torgo_root = "/home/work/Aditi/diag_paper/TORGO_database"
    extractor = TORGOTranscriptExtractor(torgo_root)

    # Prepare subset for testing
    test_speakers = ['F01', 'FC01']  # One dysarthric, one control

    output_audio = "./mfa_test/audio"
    output_transcripts = "./mfa_test/transcripts"

    print("\nPreparing TORGO corpus for MFA...")
    num_files = extractor.prepare_corpus_for_mfa(
        test_speakers,
        output_audio,
        output_transcripts
    )

    print(f"\nReady to run MFA alignment on {num_files} files")
    print("\nTo align, run:")
    print(f"  python -c \"from utils.forced_aligner import MontrealForcedAligner; "
          f"mfa = MontrealForcedAligner(); "
          f"mfa.align_corpus('{output_audio}', '{output_transcripts}', './mfa_test/output')\"")
