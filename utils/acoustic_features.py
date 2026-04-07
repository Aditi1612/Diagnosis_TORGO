"""
Acoustic Feature Extraction for DeepThink-Speech
Extracts clinically-relevant features from high-attention speech regions

Features extracted:
1. Articulatory precision: Formants (F1, F2, F3) and transitions
2. Voice quality: Jitter, Shimmer, HNR (Harmonics-to-Noise Ratio)
3. Temporal: Speaking rate, pause statistics, articulation rate
4. Spectral: Spectral tilt, centroid, zero-crossing rate
5. Phoneme-level: Using Montreal Forced Aligner (MFA) for precise alignment
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class AcousticFeatureExtractor:
    """Extract comprehensive acoustic features for clinical explanation"""

    def __init__(self, sample_rate=16000, use_mfa=False):
        """
        Initialize feature extractor

        Args:
            sample_rate: Audio sample rate
            use_mfa: Whether to use Montreal Forced Aligner for phoneme-level analysis
        """
        self.sample_rate = sample_rate
        self.use_mfa = use_mfa
        self.mfa = None

        if use_mfa:
            try:
                from .forced_aligner import MontrealForcedAligner
                self.mfa = MontrealForcedAligner()
                print("MFA initialized for phoneme-level analysis")
            except Exception as e:
                print(f"Warning: Could not initialize MFA: {e}")
                print("Continuing without phoneme-level analysis")
                self.use_mfa = False

    def extract_all_features(self, audio, high_attention_mask=None, frame_length=512, hop_length=160,
                            transcript: Optional[str] = None, audio_path: Optional[str] = None,
                            attention_weights: Optional[np.ndarray] = None):
        """
        Extract all acoustic features from audio

        Args:
            audio: Audio signal (numpy array)
            high_attention_mask: Boolean mask indicating high-attention frames
            frame_length: Frame length for analysis
            hop_length: Hop length for frame-wise analysis
            transcript: Text transcript for MFA alignment (optional)
            audio_path: Path to audio file for MFA (optional)
            attention_weights: Attention weights for phoneme-level mapping (optional)

        Returns:
            Dictionary of acoustic features
        """
        features = {}

        # If high attention mask is provided, focus on those regions
        if high_attention_mask is not None:
            audio_segments = self._extract_high_attention_audio(
                audio, high_attention_mask, hop_length
            )
            features['high_attention_segments'] = len(audio_segments)
        else:
            audio_segments = [audio]

        # Extract features from each segment
        all_segment_features = []
        for segment in audio_segments:
            if len(segment) < frame_length:
                continue

            segment_features = {}

            # 1. Formant features (articulatory precision)
            segment_features.update(self.extract_formants(segment))

            # 2. Voice quality features
            segment_features.update(self.extract_voice_quality(segment))

            # 3. Temporal features
            segment_features.update(self.extract_temporal_features(segment))

            # 4. Spectral features
            segment_features.update(self.extract_spectral_features(segment))

            # 5. Prosodic features
            segment_features.update(self.extract_prosody(segment))

            all_segment_features.append(segment_features)

        # Aggregate features across segments
        features.update(self._aggregate_features(all_segment_features))

        # 6. Phoneme-level analysis using MFA (if available)
        if self.use_mfa and self.mfa and transcript and audio_path and attention_weights is not None:
            try:
                phoneme_features = self.extract_phoneme_level_features(
                    audio_path, transcript, attention_weights, hop_length
                )
                features.update(phoneme_features)
            except Exception as e:
                print(f"Warning: Phoneme-level analysis failed: {e}")

        return features

    def _extract_high_attention_audio(self, audio, mask, hop_length):
        """Extract audio segments corresponding to high-attention frames"""
        segments = []
        in_segment = False
        segment_start = 0

        for i, is_high in enumerate(mask):
            sample_idx = i * hop_length

            if is_high and not in_segment:
                # Start new segment
                segment_start = sample_idx
                in_segment = True
            elif not is_high and in_segment:
                # End current segment
                segments.append(audio[segment_start:sample_idx])
                in_segment = False

        # Handle case where segment extends to end
        if in_segment:
            segments.append(audio[segment_start:])

        return segments

    def extract_formants(self, audio):
        """
        Extract formant frequencies (F1, F2, F3) and transitions

        Formants are crucial for assessing articulation precision in dysarthria
        """
        try:
            # Create Praat Sound object
            snd = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)

            # Extract formants
            formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)

            # Get formant values at regular intervals
            times = np.linspace(0, snd.duration, num=min(50, int(snd.duration * 100)))
            f1_values = []
            f2_values = []
            f3_values = []

            for t in times:
                try:
                    f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                    f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                    f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")

                    if not np.isnan(f1):
                        f1_values.append(f1)
                    if not np.isnan(f2):
                        f2_values.append(f2)
                    if not np.isnan(f3):
                        f3_values.append(f3)
                except:
                    continue

            # Calculate formant statistics
            features = {}

            if f1_values:
                features['F1_mean'] = np.mean(f1_values)
                features['F1_std'] = np.std(f1_values)
                features['F1_range'] = np.max(f1_values) - np.min(f1_values)
                features['F1_velocity'] = np.mean(np.abs(np.diff(f1_values)))  # Transition speed

            if f2_values:
                features['F2_mean'] = np.mean(f2_values)
                features['F2_std'] = np.std(f2_values)
                features['F2_range'] = np.max(f2_values) - np.min(f2_values)
                features['F2_velocity'] = np.mean(np.abs(np.diff(f2_values)))

            if f3_values:
                features['F3_mean'] = np.mean(f3_values)
                features['F3_std'] = np.std(f3_values)
                features['F3_range'] = np.max(f3_values) - np.min(f3_values)
                features['F3_velocity'] = np.mean(np.abs(np.diff(f3_values)))

            # Vowel space area (F1 vs F2)
            if f1_values and f2_values:
                features['vowel_space_area'] = self._calculate_vowel_space_area(f1_values, f2_values)

            return features

        except Exception as e:
            return self._default_formant_features()

    def extract_voice_quality(self, audio):
        """
        Extract voice quality features: jitter, shimmer, HNR

        These indicate voice stability and are affected in dysarthria
        """
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)

            # Pitch tracking
            pitch = call(snd, "To Pitch", 0.0, 75, 500)

            # Point process for jitter/shimmer
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

            features = {}

            # Jitter (pitch period variability)
            try:
                features['jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                features['jitter_rap'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
                features['jitter_ppq5'] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            except:
                features['jitter_local'] = 0.0
                features['jitter_rap'] = 0.0
                features['jitter_ppq5'] = 0.0

            # Shimmer (amplitude variability)
            try:
                features['shimmer_local'] = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['shimmer_apq3'] = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['shimmer_apq5'] = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            except:
                features['shimmer_local'] = 0.0
                features['shimmer_apq3'] = 0.0
                features['shimmer_apq5'] = 0.0

            # Harmonics-to-Noise Ratio
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            try:
                features['HNR_mean'] = call(harmonicity, "Get mean", 0, 0)
                features['HNR_std'] = call(harmonicity, "Get standard deviation", 0, 0)
            except:
                features['HNR_mean'] = 0.0
                features['HNR_std'] = 0.0

            # Pitch statistics
            try:
                features['pitch_mean'] = call(pitch, "Get mean", 0, 0, "Hertz")
                features['pitch_std'] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
                features['pitch_range'] = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic") - \
                                         call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            except:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0

            return features

        except Exception as e:
            return self._default_voice_quality_features()

    def extract_temporal_features(self, audio):
        """
        Extract temporal characteristics: speaking rate, pauses, articulation rate
        """
        features = {}

        # Duration
        duration = len(audio) / self.sample_rate
        features['duration'] = duration

        # Voice activity detection
        intervals = librosa.effects.split(audio, top_db=20)
        speech_duration = np.sum([end - start for start, end in intervals]) / self.sample_rate
        pause_duration = duration - speech_duration

        features['speech_duration'] = speech_duration
        features['pause_duration'] = pause_duration
        features['speech_to_pause_ratio'] = speech_duration / max(pause_duration, 0.001)

        # Speaking rate (approximate syllables per second)
        # Using energy peaks as proxy for syllables
        energy = librosa.feature.rms(y=audio)[0]
        peaks, _ = signal.find_peaks(energy, distance=int(self.sample_rate * 0.1 / 512))
        features['speaking_rate_sps'] = len(peaks) / duration  # syllables per second

        # Pause statistics
        features['num_pauses'] = len(intervals) - 1
        if len(intervals) > 1:
            pause_lengths = [(intervals[i+1][0] - intervals[i][1]) / self.sample_rate
                           for i in range(len(intervals) - 1)]
            features['mean_pause_duration'] = np.mean(pause_lengths)
            features['std_pause_duration'] = np.std(pause_lengths)
        else:
            features['mean_pause_duration'] = 0.0
            features['std_pause_duration'] = 0.0

        # Articulation rate (speech rate excluding pauses)
        features['articulation_rate'] = len(peaks) / max(speech_duration, 0.001)

        return features

    def extract_spectral_features(self, audio):
        """
        Extract spectral characteristics
        """
        features = {}

        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        # Spectral flatness (tonality)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_band{i}_mean'] = np.mean(spectral_contrast[i])

        return features

    def extract_prosody(self, audio):
        """Extract prosodic features (intonation, stress)"""
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            pitch = call(snd, "To Pitch", 0.0, 75, 500)
            intensity = call(snd, "To Intensity", 75, 0.0, "yes")

            features = {}

            # Pitch variation (intonation)
            pitch_values = []
            times = np.linspace(0, snd.duration, num=min(50, int(snd.duration * 100)))
            for t in times:
                try:
                    p = call(pitch, "Get value at time", t, "Hertz", "Linear")
                    if not np.isnan(p) and p > 0:
                        pitch_values.append(p)
                except:
                    continue

            if pitch_values:
                features['pitch_cv'] = np.std(pitch_values) / np.mean(pitch_values)  # Coefficient of variation
                features['pitch_skewness'] = skew(pitch_values)
                features['pitch_kurtosis'] = kurtosis(pitch_values)

            # Intensity variation
            intensity_values = []
            for t in times:
                try:
                    i = call(intensity, "Get value at time", t, "Cubic")
                    if not np.isnan(i):
                        intensity_values.append(i)
                except:
                    continue

            if intensity_values:
                features['intensity_mean'] = np.mean(intensity_values)
                features['intensity_std'] = np.std(intensity_values)
                features['intensity_range'] = np.max(intensity_values) - np.min(intensity_values)

            return features

        except:
            return {}

    def _calculate_vowel_space_area(self, f1_values, f2_values):
        """Calculate vowel space area (convex hull of F1-F2 space)"""
        try:
            from scipy.spatial import ConvexHull
            points = np.column_stack([f1_values, f2_values])
            hull = ConvexHull(points)
            return hull.volume  # In 2D, volume is area
        except:
            return 0.0

    def _aggregate_features(self, segment_features_list):
        """Aggregate features across segments"""
        if not segment_features_list:
            return {}

        aggregated = {}
        all_keys = set()
        for seg_feat in segment_features_list:
            all_keys.update(seg_feat.keys())

        for key in all_keys:
            values = [seg[key] for seg in segment_features_list if key in seg]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values) if len(values) > 1 else 0.0
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)

        return aggregated

    def _default_formant_features(self):
        """Default formant features when extraction fails"""
        return {
            'F1_mean': 0.0, 'F1_std': 0.0, 'F1_range': 0.0, 'F1_velocity': 0.0,
            'F2_mean': 0.0, 'F2_std': 0.0, 'F2_range': 0.0, 'F2_velocity': 0.0,
            'F3_mean': 0.0, 'F3_std': 0.0, 'F3_range': 0.0, 'F3_velocity': 0.0,
            'vowel_space_area': 0.0
        }

    def _default_voice_quality_features(self):
        """Default voice quality features when extraction fails"""
        return {
            'jitter_local': 0.0, 'jitter_rap': 0.0, 'jitter_ppq5': 0.0,
            'shimmer_local': 0.0, 'shimmer_apq3': 0.0, 'shimmer_apq5': 0.0,
            'HNR_mean': 0.0, 'HNR_std': 0.0,
            'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_range': 0.0
        }

    def extract_phoneme_level_features(self, audio_path: str, transcript: str,
                                      attention_weights: np.ndarray, hop_length: int = 160) -> Dict:
        """
        Extract phoneme-level features using Montreal Forced Aligner

        Args:
            audio_path: Path to audio file
            transcript: Text transcript
            attention_weights: Attention weights from model
            hop_length: Hop length used for MFCC extraction

        Returns:
            Dictionary with phoneme-level analysis
        """
        if not self.use_mfa or not self.mfa:
            return {}

        try:
            # Load audio to get duration
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            audio_duration = len(audio) / sr

            # Perform forced alignment
            alignment = self.mfa.align_single_file(audio_path, transcript)

            # Map attention weights to phonemes
            phoneme_attention = self.mfa.map_attention_to_phonemes(
                alignment,
                attention_weights,
                audio_duration,
                hop_length,
                self.sample_rate
            )

            # Analyze phoneme classes
            class_analysis = self.mfa.analyze_phoneme_classes(phoneme_attention)

            # Identify most affected phonemes (high attention = likely problematic)
            affected_phonemes = phoneme_attention['top_phonemes'][:5]

            features = {
                'phoneme_alignment_available': True,
                'num_phonemes': len(alignment['phones']),
                'most_attended_phonemes': affected_phonemes,
                'phoneme_class_attention': class_analysis['class_attention'],
                'highest_attention_class': class_analysis['highest_attention_class'],
                'phoneme_class_counts': class_analysis['class_counts'],
                'phoneme_attention_scores': phoneme_attention['phoneme_attention']
            }

            return features

        except Exception as e:
            print(f"Error in phoneme-level feature extraction: {e}")
            return {'phoneme_alignment_available': False}


if __name__ == '__main__':
    # Test acoustic feature extraction
    extractor = AcousticFeatureExtractor(sample_rate=16000)

    # Create dummy audio
    duration = 2.0  # seconds
    sample_rate = 16000
    audio = np.random.randn(int(duration * sample_rate)) * 0.1

    # Extract features
    features = extractor.extract_all_features(audio)

    print("Extracted acoustic features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
