"""
LLM-based Explanation Generation for DeepThink-Speech
Stage 2: Leveraging DeepSeek-R1 for Natural Language Clinical Explanations

Transforms attention patterns and acoustic features into clinically-interpretable explanations
"""

import json
import numpy as np
from typing import Dict, List, Optional
import torch


class ClinicalExplainer:
    """
    Generate clinical explanations using LLM reasoning
    Integrates with DeepSeek-R1-Distill-14B or similar reasoning-capable LLMs
    """

    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device="cuda"):
        """
        Initialize the clinical explainer

        Args:
            model_name: HuggingFace model name or path
            device: Device for model inference
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

        # Clinical reference ranges for dysarthria assessment
        self.reference_ranges = {
            'formants': {
                'F1': {'normal_mean': (500, 800), 'dysarthric_mean': (400, 900)},
                'F2': {'normal_mean': (1400, 2200), 'dysarthric_mean': (1200, 2400)},
                'F3': {'normal_mean': (2500, 3500), 'dysarthric_mean': (2000, 3800)},
            },
            'voice_quality': {
                'jitter': {'normal': (0.0, 1.0), 'mild': (1.0, 2.5), 'severe': (2.5, float('inf'))},
                'shimmer': {'normal': (0.0, 3.5), 'mild': (3.5, 7.0), 'severe': (7.0, float('inf'))},
                'HNR': {'normal': (20, float('inf')), 'mild': (15, 20), 'severe': (0, 15)},
            },
            'temporal': {
                'speaking_rate': {'normal': (4.0, 6.0), 'slow': (0, 4.0), 'fast': (6.0, float('inf'))},
                'articulation_rate': {'normal': (4.5, 6.5), 'slow': (0, 4.5), 'fast': (6.5, float('inf'))},
            }
        }

    def load_model(self):
        """Load the LLM model for explanation generation"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model. Using rule-based explanation. Error: {e}")
            self.model = None

    def generate_explanation(
        self,
        prediction: int,
        confidence: float,
        attention_weights: np.ndarray,
        acoustic_features: Dict,
        audio_metadata: Optional[Dict] = None,
        explanation_level: str = "clinical"
    ) -> Dict:
        """
        Generate comprehensive explanation for dysarthria detection

        Args:
            prediction: Model prediction (0: healthy, 1: dysarthric)
            confidence: Prediction confidence score
            attention_weights: Attention weight matrix
            acoustic_features: Dictionary of extracted acoustic features
            audio_metadata: Optional metadata (speaker info, utterance, etc.)
            explanation_level: "clinical" for SLPs, "patient" for patients/families

        Returns:
            Dictionary containing explanation components
        """
        # Analyze attention patterns
        attention_analysis = self._analyze_attention_patterns(attention_weights)

        # Analyze acoustic features
        acoustic_analysis = self._analyze_acoustic_features(acoustic_features)

        # Create prompt for LLM
        prompt = self._create_explanation_prompt(
            prediction=prediction,
            confidence=confidence,
            attention_analysis=attention_analysis,
            acoustic_analysis=acoustic_analysis,
            audio_metadata=audio_metadata,
            explanation_level=explanation_level
        )

        # Generate explanation using LLM (if loaded) or rule-based system
        if self.model is not None:
            explanation_text = self._generate_with_llm(prompt)
        else:
            explanation_text = self._generate_rule_based_explanation(
                prediction, confidence, attention_analysis, acoustic_analysis, explanation_level
            )

        # Structure the complete explanation
        explanation = {
            'prediction': 'Dysarthric' if prediction == 1 else 'Healthy',
            'confidence': confidence,
            'explanation': explanation_text,
            'attention_analysis': attention_analysis,
            'acoustic_analysis': acoustic_analysis,
            'clinical_indicators': self._extract_clinical_indicators(acoustic_analysis),
            'recommendations': self._generate_recommendations(prediction, acoustic_analysis)
        }

        return explanation

    def _analyze_attention_patterns(self, attention_weights: np.ndarray) -> Dict:
        """Analyze attention weight patterns"""
        # Average across heads and query positions
        if len(attention_weights.shape) == 4:  # (batch, heads, seq, seq)
            attention_scores = attention_weights[0].mean(axis=0).mean(axis=0)
        else:
            attention_scores = attention_weights

        # Find high-attention regions
        threshold = np.percentile(attention_scores, 75)
        high_attention_frames = np.where(attention_scores >= threshold)[0]

        # Compute statistics
        analysis = {
            'high_attention_frames': high_attention_frames.tolist(),
            'num_high_attention_regions': len(high_attention_frames),
            'attention_concentration': float(np.max(attention_scores) / (np.mean(attention_scores) + 1e-8)),
            'attention_distribution': {
                'mean': float(np.mean(attention_scores)),
                'std': float(np.std(attention_scores)),
                'max': float(np.max(attention_scores)),
                'min': float(np.min(attention_scores))
            },
            'temporal_focus': self._identify_temporal_focus(attention_scores)
        }

        return analysis

    def _identify_temporal_focus(self, attention_scores: np.ndarray) -> str:
        """Identify where in the utterance the model focused most"""
        third = len(attention_scores) // 3
        beginning = np.mean(attention_scores[:third])
        middle = np.mean(attention_scores[third:2*third])
        end = np.mean(attention_scores[2*third:])

        focus_region = max([('beginning', beginning), ('middle', middle), ('end', end)],
                          key=lambda x: x[1])

        return focus_region[0]

    def _analyze_acoustic_features(self, features: Dict) -> Dict:
        """Analyze acoustic features against clinical norms"""
        analysis = {
            'articulatory_analysis': self._analyze_articulatory_features(features),
            'voice_quality_analysis': self._analyze_voice_quality(features),
            'temporal_analysis': self._analyze_temporal_features(features),
            'overall_severity': 'normal'  # Will be determined based on features
        }

        # Determine overall severity
        severity_scores = []
        for category_analysis in [analysis['articulatory_analysis'],
                                 analysis['voice_quality_analysis'],
                                 analysis['temporal_analysis']]:
            if category_analysis.get('severity'):
                severity_scores.append(category_analysis['severity'])

        if 'severe' in severity_scores:
            analysis['overall_severity'] = 'severe'
        elif 'moderate' in severity_scores:
            analysis['overall_severity'] = 'moderate'
        elif 'mild' in severity_scores:
            analysis['overall_severity'] = 'mild'

        return analysis

    def _analyze_articulatory_features(self, features: Dict) -> Dict:
        """Analyze formants and articulatory precision"""
        analysis = {'findings': [], 'severity': 'normal'}

        # Analyze F1, F2, F3
        for formant in ['F1', 'F2', 'F3']:
            mean_key = f'{formant}_mean_mean'
            std_key = f'{formant}_std_mean'
            velocity_key = f'{formant}_velocity_mean'

            if mean_key in features:
                f_mean = features[mean_key]
                normal_range = self.reference_ranges['formants'][formant]['normal_mean']

                # Check if within normal range
                if f_mean < normal_range[0] * 0.8 or f_mean > normal_range[1] * 1.2:
                    analysis['findings'].append(f"{formant} frequency abnormal: {f_mean:.1f} Hz")
                    analysis['severity'] = 'moderate'

            if velocity_key in features and features[velocity_key] > 100:
                analysis['findings'].append(f"{formant} transition velocity elevated: {features[velocity_key]:.1f} Hz/frame")

        # Vowel space area
        if 'vowel_space_area_mean' in features:
            vsa = features['vowel_space_area_mean']
            if vsa < 50000:  # Reduced vowel space indicates articulatory deficit
                analysis['findings'].append(f"Reduced vowel space area: {vsa:.0f}")
                analysis['severity'] = 'moderate'

        return analysis

    def _analyze_voice_quality(self, features: Dict) -> Dict:
        """Analyze jitter, shimmer, HNR"""
        analysis = {'findings': [], 'severity': 'normal'}

        # Jitter
        if 'jitter_local_mean' in features:
            jitter = features['jitter_local_mean'] * 100  # Convert to percentage
            if jitter > self.reference_ranges['voice_quality']['jitter']['severe'][0]:
                analysis['findings'].append(f"Severe jitter: {jitter:.2f}%")
                analysis['severity'] = 'severe'
            elif jitter > self.reference_ranges['voice_quality']['jitter']['mild'][0]:
                analysis['findings'].append(f"Elevated jitter: {jitter:.2f}%")
                analysis['severity'] = 'mild'

        # Shimmer
        if 'shimmer_local_mean' in features:
            shimmer = features['shimmer_local_mean'] * 100
            if shimmer > self.reference_ranges['voice_quality']['shimmer']['severe'][0]:
                analysis['findings'].append(f"Severe shimmer: {shimmer:.2f}%")
                analysis['severity'] = 'severe'
            elif shimmer > self.reference_ranges['voice_quality']['shimmer']['mild'][0]:
                analysis['findings'].append(f"Elevated shimmer: {shimmer:.2f}%")
                if analysis['severity'] == 'normal':
                    analysis['severity'] = 'mild'

        # HNR
        if 'HNR_mean_mean' in features:
            hnr = features['HNR_mean_mean']
            if hnr < self.reference_ranges['voice_quality']['HNR']['severe'][1]:
                analysis['findings'].append(f"Low harmonics-to-noise ratio: {hnr:.1f} dB")
                analysis['severity'] = 'severe'
            elif hnr < self.reference_ranges['voice_quality']['HNR']['mild'][1]:
                analysis['findings'].append(f"Reduced harmonics-to-noise ratio: {hnr:.1f} dB")
                if analysis['severity'] == 'normal':
                    analysis['severity'] = 'mild'

        return analysis

    def _analyze_temporal_features(self, features: Dict) -> Dict:
        """Analyze speaking rate and temporal patterns"""
        analysis = {'findings': [], 'severity': 'normal'}

        # Speaking rate
        if 'speaking_rate_sps_mean' in features:
            rate = features['speaking_rate_sps_mean']
            if rate < self.reference_ranges['temporal']['speaking_rate']['slow'][1]:
                analysis['findings'].append(f"Reduced speaking rate: {rate:.1f} syllables/sec")
                analysis['severity'] = 'moderate'
            elif rate > self.reference_ranges['temporal']['speaking_rate']['fast'][0]:
                analysis['findings'].append(f"Accelerated speaking rate: {rate:.1f} syllables/sec")

        # Pause patterns
        if 'mean_pause_duration_mean' in features:
            pause_dur = features['mean_pause_duration_mean']
            if pause_dur > 1.0:  # Long pauses
                analysis['findings'].append(f"Extended pauses: {pause_dur:.2f} sec average")
                analysis['severity'] = 'mild'

        return analysis

    def _create_explanation_prompt(
        self,
        prediction: int,
        confidence: float,
        attention_analysis: Dict,
        acoustic_analysis: Dict,
        audio_metadata: Optional[Dict],
        explanation_level: str
    ) -> str:
        """Create prompt for LLM explanation generation"""

        diagnosis = "dysarthric speech" if prediction == 1 else "healthy speech"

        # Check for phoneme-level analysis
        has_phoneme_analysis = acoustic_analysis.get('phoneme_alignment_available', False)

        prompt = f"""You are an expert speech-language pathologist providing clinical explanations for an AI-assisted dysarthria detection system.

<think>
Analyze the following diagnostic information systematically:

**Model Prediction**: {diagnosis} (confidence: {confidence:.1%})

**Attention Analysis**:
- The model focused on {attention_analysis['num_high_attention_regions']} key temporal regions
- Primary temporal focus: {attention_analysis['temporal_focus']} of the utterance
- Attention concentration: {attention_analysis['attention_concentration']:.2f}

**Acoustic Analysis**:

1. Articulatory Features:
{self._format_findings(acoustic_analysis['articulatory_analysis'])}

2. Voice Quality:
{self._format_findings(acoustic_analysis['voice_quality_analysis'])}

3. Temporal Patterns:
{self._format_findings(acoustic_analysis['temporal_analysis'])}
"""

        # Add phoneme-level analysis if available
        if has_phoneme_analysis:
            phoneme_info = self._format_phoneme_analysis(acoustic_analysis)
            prompt += f"""
4. Phoneme-Level Analysis (Montreal Forced Aligner):
{phoneme_info}
"""

        prompt += f"""
**Overall Severity Assessment**: {acoustic_analysis['overall_severity']}

Step-by-step reasoning:
1. What specific acoustic abnormalities were detected?
2. How do these abnormalities relate to neuromotor impairments?"""

        if has_phoneme_analysis:
            prompt += "\n3. Which specific phonemes or phoneme classes show the most impairment?"
            prompt += "\n4. Which dysarthria subtypes do these patterns suggest?"
            prompt += "\n5. What clinical actions should be recommended?"
        else:
            prompt += "\n3. Which dysarthria subtypes do these patterns suggest?"
            prompt += "\n4. What clinical actions should be recommended?"

        prompt += f"""
</think>

Generate a {"clinical report for speech-language pathologists" if explanation_level == "clinical" else "patient-friendly explanation"}:
"""

        return prompt

    def _format_findings(self, analysis: Dict) -> str:
        """Format findings for prompt"""
        if not analysis['findings']:
            return "- No significant abnormalities detected"
        return "\n".join([f"- {finding}" for finding in analysis['findings']])

    def _format_phoneme_analysis(self, acoustic_analysis: Dict) -> str:
        """Format phoneme-level analysis for prompt"""
        if not acoustic_analysis.get('phoneme_alignment_available'):
            return "- Phoneme-level analysis not available"

        formatted = []

        # Most attended phonemes (likely problematic)
        if 'most_attended_phonemes' in acoustic_analysis:
            phonemes = ", ".join(acoustic_analysis['most_attended_phonemes'])
            formatted.append(f"- Most affected phonemes: {phonemes}")

        # Phoneme class analysis
        if 'highest_attention_class' in acoustic_analysis:
            highest_class = acoustic_analysis['highest_attention_class']
            formatted.append(f"- Phoneme class most affected: {highest_class}")

        # Class attention details
        if 'phoneme_class_attention' in acoustic_analysis:
            class_attn = acoustic_analysis['phoneme_class_attention']
            sorted_classes = sorted(class_attn.items(), key=lambda x: x[1], reverse=True)[:3]
            class_str = ", ".join([f"{cls}: {attn:.3f}" for cls, attn in sorted_classes])
            formatted.append(f"- Attention by class: {class_str}")

        if not formatted:
            return "- Phoneme-level analysis completed but no specific findings"

        return "\n".join(formatted)

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate explanation using LLM"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part (after prompt)
            explanation = explanation[len(prompt):].strip()
            return explanation
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return "Explanation generation failed. Using rule-based explanation."

    def _generate_rule_based_explanation(
        self,
        prediction: int,
        confidence: float,
        attention_analysis: Dict,
        acoustic_analysis: Dict,
        explanation_level: str
    ) -> str:
        """Generate rule-based explanation when LLM is unavailable"""

        diagnosis = "dysarthric" if prediction == 1 else "healthy"
        severity = acoustic_analysis['overall_severity']

        if explanation_level == "clinical":
            explanation = f"""**Clinical Assessment Report**

**Diagnosis**: {diagnosis.capitalize()} speech (confidence: {confidence:.1%})
**Severity**: {severity.capitalize()}

**Temporal Focus**: The model concentrated attention on the {attention_analysis['temporal_focus']} portion of the utterance, analyzing {attention_analysis['num_high_attention_regions']} discriminative temporal regions.

**Acoustic Findings**:

*Articulatory Precision*:
{self._format_findings(acoustic_analysis['articulatory_analysis'])}

*Voice Quality*:
{self._format_findings(acoustic_analysis['voice_quality_analysis'])}

*Temporal Characteristics*:
{self._format_findings(acoustic_analysis['temporal_analysis'])}

**Clinical Interpretation**:
The observed acoustic abnormalities are consistent with {diagnosis} speech patterns. These features suggest potential neuromotor impairments affecting speech production subsystems.
"""
        else:  # patient-friendly
            explanation = f"""**Speech Assessment Summary**

Based on the analysis, the speech sample shows signs of {diagnosis} speech patterns (confidence level: {confidence:.1%}).

**What We Found**:
The AI system carefully analyzed different parts of the speech recording and identified specific patterns that indicate {diagnosis} speech.

**Key Observations**:
"""
            # Simplify findings for patients
            all_findings = (acoustic_analysis['articulatory_analysis']['findings'] +
                          acoustic_analysis['voice_quality_analysis']['findings'] +
                          acoustic_analysis['temporal_analysis']['findings'])

            if all_findings:
                for finding in all_findings[:3]:  # Top 3 findings
                    explanation += f"- {finding}\n"
            else:
                explanation += "- Speech patterns within normal ranges\n"

        return explanation

    def _extract_clinical_indicators(self, acoustic_analysis: Dict) -> List[str]:
        """Extract key clinical indicators for summary"""
        indicators = []

        for category in ['articulatory_analysis', 'voice_quality_analysis', 'temporal_analysis']:
            findings = acoustic_analysis[category]['findings']
            indicators.extend(findings)

        return indicators

    def _generate_recommendations(self, prediction: int, acoustic_analysis: Dict) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []

        if prediction == 0:
            return ["No intervention required. Continue routine monitoring."]

        severity = acoustic_analysis['overall_severity']

        if severity in ['moderate', 'severe']:
            recommendations.append("Recommend comprehensive speech-language pathology evaluation")

        if acoustic_analysis['articulatory_analysis']['findings']:
            recommendations.append("Consider articulatory therapy targeting formant production")

        if acoustic_analysis['voice_quality_analysis']['findings']:
            recommendations.append("Evaluate for voice therapy to address phonatory stability")

        if acoustic_analysis['temporal_analysis']['findings']:
            recommendations.append("Assess rate control and pacing strategies")

        return recommendations


if __name__ == '__main__':
    # Test the explainer with dummy data
    explainer = ClinicalExplainer()

    # Dummy data
    prediction = 1
    confidence = 0.87
    attention_weights = np.random.rand(1, 8, 100, 100)  # (batch, heads, seq, seq)
    acoustic_features = {
        'F1_mean_mean': 650.0,
        'F2_mean_mean': 1800.0,
        'jitter_local_mean': 0.025,
        'shimmer_local_mean': 0.055,
        'HNR_mean_mean': 18.5,
        'speaking_rate_sps_mean': 3.2
    }

    # Generate explanation
    explanation = explainer.generate_explanation(
        prediction=prediction,
        confidence=confidence,
        attention_weights=attention_weights,
        acoustic_features=acoustic_features,
        explanation_level="clinical"
    )

    print("Generated Explanation:")
    print(json.dumps(explanation, indent=2))
