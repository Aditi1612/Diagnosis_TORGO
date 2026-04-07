#!/bin/bash
# Ablation Study 2: Pooling Strategy Comparison
# Compares: mean vs attention vs weighted-sum vs max vs first

echo "=============================================="
echo "ABLATION STUDY 2: Pooling Strategy Comparison"
echo "=============================================="

OUTPUT_BASE="./output/ablation_pooling"
DATA_ROOT="./TORGO_database"
EPOCHS=30
BATCH_SIZE=8
SEED=42

# Create results directory
mkdir -p ${OUTPUT_BASE}/results

# ============================================
# 1. Mean Pooling (Baseline)
# ============================================
echo ""
echo "[1/5] Training with Mean Pooling..."
echo "=============================================="

python scripts/train_ablation.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_BASE}/mean \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${EPOCHS} \
    --max_length_sec 4.0 \
    --seed ${SEED} \
    --mode both

# ============================================
# 2. Attention Pooling
# ============================================
echo ""
echo "[2/5] Training with Attention Pooling..."
echo "=============================================="

python scripts/train_ablation.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_BASE}/attention \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling attention \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${EPOCHS} \
    --max_length_sec 4.0 \
    --seed ${SEED} \
    --mode both

# ============================================
# 3. Weighted-Sum Pooling (across layers)
# ============================================
echo ""
echo "[3/5] Training with Weighted-Sum Pooling..."
echo "=============================================="

python scripts/train_ablation.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_BASE}/weighted_sum \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling weighted_sum \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${EPOCHS} \
    --max_length_sec 4.0 \
    --seed ${SEED} \
    --mode both

# ============================================
# 4. Max Pooling
# ============================================
echo ""
echo "[4/5] Training with Max Pooling..."
echo "=============================================="

python scripts/train_ablation.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_BASE}/max \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling max \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${EPOCHS} \
    --max_length_sec 4.0 \
    --seed ${SEED} \
    --mode both

# ============================================
# 5. First Token Pooling
# ============================================
echo ""
echo "[5/5] Training with First Token Pooling..."
echo "=============================================="

python scripts/train_ablation.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_BASE}/first \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling first \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${EPOCHS} \
    --max_length_sec 4.0 \
    --seed ${SEED} \
    --mode both

# ============================================
# Collect Results
# ============================================
echo ""
echo "=============================================="
echo "ABLATION STUDY 2: RESULTS SUMMARY"
echo "=============================================="

python -c "
import json
from pathlib import Path

output_base = Path('${OUTPUT_BASE}')
poolings = ['mean', 'attention', 'weighted_sum', 'max', 'first']
pooling_names = ['Mean', 'Attention', 'Weighted-Sum', 'Max', 'First Token']

results = []
print(f\"{'Pooling':<15} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Bal.Acc':>10}\")
print('='*57)

for pooling, name in zip(poolings, pooling_names):
    result_file = output_base / pooling / 'results' / 'test_results_latest.json'
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
            m = data['metrics']
            print(f\"{name:<15} {m['accuracy']*100:>9.2f}% {m['f1']:>10.4f} {m['auc']:>10.4f} {m['balanced_accuracy']*100:>9.2f}%\")
            results.append({'pooling': name, **m})
    else:
        print(f\"{name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}\")

# Save combined results
with open(output_base / 'results' / 'ablation_pooling_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f\"\nResults saved to: {output_base}/results/ablation_pooling_results.json\")
"

echo ""
echo "Ablation Study 2 Complete!"
