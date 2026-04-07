#!/usr/bin/env python3
"""
Verify Stage 2 metrics from test_explanations.json
"""

import json
import numpy as np
from pathlib import Path

# Load results
results_file = Path("output/stage2_explanations/test_explanations.json")
with open(results_file, 'r') as f:
    data = json.load(f)

print("=" * 70)
print("STAGE 2 METRICS VERIFICATION")
print("=" * 70)

# Basic stats
total = len(data)
correct = sum(1 for d in data if d['prediction'] == d['ground_truth'])
confidences = [d['confidence'] for d in data]

print(f"\n1. CLASSIFICATION PERFORMANCE")
print(f"   Total samples: {total}")
print(f"   Correct: {correct} ({correct/total*100:.2f}%)")
print(f"   Avg confidence: {np.mean(confidences):.4f} ± {np.std(confidences):.4f}")
print(f"   Confidence range: [{np.min(confidences):.4f}, {np.max(confidences):.4f}]")

# Attention faithfulness
faithfulness = [d['attention_analysis']['attention_concentration'] for d in data]
print(f"\n2. ATTENTION FAITHFULNESS")
print(f"   Mean concentration: {np.mean(faithfulness):.4f} ± {np.std(faithfulness):.4f}")

# High attention regions
high_attn_regions = [d['attention_analysis']['num_high_attention_regions'] for d in data]
print(f"\n3. ATTENTION REGIONS")
print(f"   Avg high-attention regions: {np.mean(high_attn_regions):.1f} ± {np.std(high_attn_regions):.1f}")
print(f"   Range: [{np.min(high_attn_regions)}, {np.max(high_attn_regions)}]")

# Temporal focus distribution
temporal_focus = [d['attention_analysis']['temporal_focus'] for d in data]
focus_counts = {'beginning': 0, 'middle': 0, 'end': 0}
for focus in temporal_focus:
    if focus in focus_counts:
        focus_counts[focus] += 1

print(f"\n4. TEMPORAL FOCUS DISTRIBUTION")
for focus, count in focus_counts.items():
    print(f"   {focus.capitalize()}: {count}/{total} ({count/total*100:.1f}%)")

# Severity distribution
severity = [d['acoustic_analysis']['overall_severity'] for d in data]
severity_counts = {}
for s in severity:
    severity_counts[s] = severity_counts.get(s, 0) + 1

print(f"\n5. SEVERITY STRATIFICATION")
for sev, count in sorted(severity_counts.items()):
    print(f"   {sev.capitalize()}: {count}/{total} ({count/total*100:.1f}%)")

# Explanation length (count tokens in explanation text)
explanation_lengths = []
for d in data:
    if 'explanation' in d:
        # Simple whitespace tokenization
        tokens = d['explanation'].split()
        explanation_lengths.append(len(tokens))

print(f"\n6. EXPLANATION LENGTH")
print(f"   Avg length: {np.mean(explanation_lengths):.1f} ± {np.std(explanation_lengths):.1f} tokens")
print(f"   Range: [{np.min(explanation_lengths)}, {np.max(explanation_lengths)}]")

# Recommendations count
recommendation_counts = []
for d in data:
    if 'recommendations' in d:
        recommendation_counts.append(len(d['recommendations']))

print(f"\n7. THERAPEUTIC RECOMMENDATIONS")
print(f"   Avg per case: {np.mean(recommendation_counts):.1f} ± {np.std(recommendation_counts):.1f}")
print(f"   Range: [{np.min(recommendation_counts)}, {np.max(recommendation_counts)}]")

# Clinical coverage
print(f"\n8. CLINICAL DOMAIN COVERAGE")
art_count = sum(1 for d in data if len(d['acoustic_analysis']['articulatory_analysis']['findings']) > 0)
voice_count = sum(1 for d in data if len(d['acoustic_analysis']['voice_quality_analysis']['findings']) > 0)
temp_count = sum(1 for d in data if len(d['acoustic_analysis']['temporal_analysis']['findings']) > 0)

print(f"   Articulatory: {art_count}/{total} ({art_count/total*100:.1f}%)")
print(f"   Voice Quality: {voice_count}/{total} ({voice_count/total*100:.1f}%)")
print(f"   Temporal: {temp_count}/{total} ({temp_count/total*100:.1f}%)")
print(f"   All domains: {min(art_count, voice_count, temp_count)}/{total}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
