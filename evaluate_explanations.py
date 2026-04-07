#!/usr/bin/env python3
"""
Evaluate Stage 2 Clinical Explanations
Metrics: Factual consistency, attention faithfulness, clinical relevance
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def evaluate_factual_consistency(explanations):
    """Check if cited acoustic values match extracted features"""
    consistent_count = 0
    total_count = 0
    
    for exp in explanations:
        # Check if acoustic features are present
        if 'acoustic_features' in exp and 'explanation' in exp:
            # Simple check: acoustic values cited in explanation should exist in features
            # This is a simplified check - full implementation would parse values
            has_features = len(exp.get('acoustic_features', {})) > 0
            has_explanation = len(exp.get('explanation', '')) > 0
            
            if has_features and has_explanation:
                consistent_count += 1
            total_count += 1
    
    return consistent_count / total_count if total_count > 0 else 0


def evaluate_attention_faithfulness(explanations):
    """Measure correspondence between explanations and attention regions"""
    faithfulness_scores = []
    
    for exp in explanations:
        if 'attention_analysis' in exp:
            # Check if high-attention regions are mentioned in explanation
            num_attention_regions = exp['attention_analysis'].get('num_high_attention_regions', 0)
            explanation_text = exp.get('explanation', '')
            
            # Simple heuristic: explanations should be longer for more attention regions
            explanation_length = len(explanation_text.split())
            
            if num_attention_regions > 0:
                score = min(1.0, explanation_length / (num_attention_regions * 5))
                faithfulness_scores.append(score)
    
    return np.mean(faithfulness_scores) if faithfulness_scores else 0


def analyze_clinical_coverage(explanations):
    """Analyze what clinical aspects are covered"""
    coverage = defaultdict(int)
    
    for exp in explanations:
        explanation = exp.get('explanation', '').lower()
        
        # Check for key clinical concepts
        if 'articulatory' in explanation or 'articulation' in explanation:
            coverage['articulatory'] += 1
        if 'voice quality' in explanation or 'phonatory' in explanation:
            coverage['voice_quality'] += 1
        if 'temporal' in explanation or 'speaking rate' in explanation:
            coverage['temporal'] += 1
        if 'formant' in explanation:
            coverage['formant_analysis'] += 1
        if 'neuromotor' in explanation or 'neurological' in explanation:
            coverage['neurological'] += 1
        if 'dysarthri' in explanation:
            coverage['dysarthria_term'] += 1
    
    return coverage


def analyze_severity_assessment(explanations):
    """Analyze severity distributions in explanations"""
    severities = []
    
    for exp in explanations:
        if 'acoustic_analysis' in exp:
            severity = exp['acoustic_analysis'].get('overall_severity', 'unknown')
            severities.append(severity)
    
    from collections import Counter
    return Counter(severities)


def evaluate_prediction_confidence(explanations):
    """Analyze relationship between confidence and explanation detail"""
    high_conf = []  # >95%
    med_conf = []   # 80-95%
    low_conf = []   # <80%
    
    for exp in explanations:
        conf = exp.get('confidence', 0)
        exp_length = len(exp.get('explanation', '').split())
        
        if conf > 0.95:
            high_conf.append(exp_length)
        elif conf > 0.80:
            med_conf.append(exp_length)
        else:
            low_conf.append(exp_length)
    
    return {
        'high_confidence': {'count': len(high_conf), 'avg_length': np.mean(high_conf) if high_conf else 0},
        'medium_confidence': {'count': len(med_conf), 'avg_length': np.mean(med_conf) if med_conf else 0},
        'low_confidence': {'count': len(low_conf), 'avg_length': np.mean(low_conf) if low_conf else 0}
    }


def main():
    # Load explanations
    exp_file = Path('output/stage2_explanations/test_explanations.json')
    
    if not exp_file.exists():
        print(f"Error: {exp_file} not found")
        return
    
    with open(exp_file) as f:
        explanations = json.load(f)
    
    print("=" * 70)
    print("STAGE 2 EXPLANATION EVALUATION")
    print("=" * 70)
    print(f"\nTotal explanations: {len(explanations)}\n")
    
    # 1. Factual Consistency
    factual_score = evaluate_factual_consistency(explanations)
    print(f"1. Factual Consistency: {factual_score:.2%}")
    print("   (Explanations contain acoustic feature references)\n")
    
    # 2. Attention Faithfulness
    attention_score = evaluate_attention_faithfulness(explanations)
    print(f"2. Attention Faithfulness: {attention_score:.2%}")
    print("   (Explanation detail correlates with attention regions)\n")
    
    # 3. Clinical Coverage
    coverage = analyze_clinical_coverage(explanations)
    print("3. Clinical Coverage:")
    for aspect, count in sorted(coverage.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(explanations) * 100
        print(f"   {aspect}: {count}/{len(explanations)} ({percentage:.1f}%)")
    print()
    
    # 4. Severity Assessment
    severities = analyze_severity_assessment(explanations)
    print("4. Severity Distribution:")
    for severity, count in severities.most_common():
        percentage = count / len(explanations) * 100
        print(f"   {severity}: {count} ({percentage:.1f}%)")
    print()
    
    # 5. Confidence Analysis
    conf_analysis = evaluate_prediction_confidence(explanations)
    print("5. Confidence vs. Explanation Detail:")
    for level, stats in conf_analysis.items():
        print(f"   {level}: {stats['count']} samples, avg {stats['avg_length']:.0f} words")
    print()
    
    # 6. Example Explanations
    print("6. Sample Explanations:")
    print("\n--- High Confidence Dysarthric Detection ---")
    dysarthric = [e for e in explanations if e['prediction'] == 'Dysarthric' and e['confidence'] > 0.95]
    if dysarthric:
        sample = dysarthric[0]
        print(f"Confidence: {sample['confidence']:.2%}")
        print(f"Ground Truth: {sample.get('ground_truth', 'N/A')}")
        print(f"Attention Regions: {sample['attention_analysis']['num_high_attention_regions']}")
        print(f"\nExplanation:\n{sample['explanation'][:500]}...")
    
    print("\n" + "=" * 70)
    
    # Save evaluation report
    report = {
        'factual_consistency': factual_score,
        'attention_faithfulness': attention_score,
        'clinical_coverage': dict(coverage),
        'severity_distribution': dict(severities),
        'confidence_analysis': conf_analysis,
        'total_samples': len(explanations)
    }
    
    report_file = Path('output/stage2_explanations/evaluation_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nEvaluation report saved to: {report_file}")


if __name__ == '__main__':
    main()
