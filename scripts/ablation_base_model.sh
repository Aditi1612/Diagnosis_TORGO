#!/bin/bash

echo "========================================="
echo "ABLATION STUDY 1: Base Model Comparison"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Models: wav2vec2-base, wav2vec2-large, hubert-base"
echo "  - Split: Speaker-independent (severity-balanced)"
echo "  - LR: 1e-4, Epochs: 30"
echo ""

OUTPUT_BASE="./output/ablation_base_model"
DATA_ROOT="./TORGO_database"

mkdir -p ${OUTPUT_BASE}/results

# ============================================
# 1. Wav2Vec2-Base
# ============================================
echo ""
echo "[1/3] Training Wav2Vec2-Base..."
echo "========================================="

python scripts/train_wav2vec2.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_BASE}/wav2vec2_base \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --batch_size 8 \
    --num_epochs 30 \
    --learning_rate 1e-4 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --max_length_sec 4.0 \
    --mode both

# ============================================
# 2. Wav2Vec2-Large
# ============================================
echo ""
echo "[2/3] Training Wav2Vec2-Large..."
echo "========================================="

python scripts/train_wav2vec2.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_BASE}/wav2vec2_large \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-large \
    --split_mode speaker \
    --pooling mean \
    --batch_size 4 \
    --num_epochs 30 \
    --learning_rate 1e-4 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --max_length_sec 4.0 \
    --mode both

# ============================================
# 3. HuBERT-Base
# ============================================
echo ""
echo "[3/3] Training HuBERT-Base..."
echo "========================================="

python scripts/train_wav2vec2.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_BASE}/hubert_base \
    --model_type hubert \
    --model_name facebook/hubert-base-ls960 \
    --split_mode speaker \
    --pooling mean \
    --batch_size 8 \
    --num_epochs 30 \
    --learning_rate 1e-4 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --max_length_sec 4.0 \
    --mode both

# ============================================
# Collect Results
# ============================================
echo ""
echo "========================================="
echo "ABLATION STUDY 1: RESULTS SUMMARY"
echo "========================================="

python -c "
import json
from pathlib import Path

output_base = Path('./output/ablation_base_model')
models = ['wav2vec2_base', 'wav2vec2_large', 'hubert_base']
model_names = ['Wav2Vec2-Base', 'Wav2Vec2-Large', 'HuBERT-Base']

results = []
print(f\"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Bal.Acc':>10}\")
print('='*62)

for model, name in zip(models, model_names):
    result_file = output_base / model / 'results' / 'test_results_latest.json'
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
            m = data['metrics']
            print(f\"{name:<20} {m['accuracy']*100:>9.2f}% {m['f1']:>10.4f} {m['auc']:>10.4f} {m['balanced_accuracy']*100:>9.2f}%\")
            results.append({'model': name, **m})
    else:
        print(f\"{name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}\")

with open(output_base / 'results' / 'ablation_base_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f\"\nResults saved to: {output_base}/results/ablation_base_model_results.json\")
"

echo ""
echo "Ablation Study 1 Complete!"
