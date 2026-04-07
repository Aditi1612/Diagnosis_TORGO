#!/bin/bash

# Activate diag_paper conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diag_paper

cd /home/work/Aditi/diag_paper/Large-Scale-Multimodal-Depression-Detection

echo "========================================="
echo "Testing Pooling Ablation Models"
echo "Using diag_paper conda environment"
echo "========================================="
echo ""

for pooling_type in mean attention max cls; do
    dir="output/ablation/pooling_${pooling_type}"
    if [ -f "$dir/checkpoints/best_model.pth" ]; then
        echo "--- Testing Pooling: $pooling_type ---"
        python scripts/train_wav2vec2.py \
            --data_root ./TORGO_database \
            --output_dir "$dir" \
            --model_type wav2vec2 \
            --model_name facebook/wav2vec2-base \
            --split_mode speaker \
            --pooling "$pooling_type" \
            --batch_size 8 \
            --max_length_sec 4.0 \
            --checkpoint "$dir/checkpoints/best_model.pth" \
            --mode test
        echo ""
    else
        echo "--- Pooling $pooling_type: Not trained yet ---"
        echo ""
    fi
done

echo "========================================="
echo "Testing Complete! Results:"
echo "========================================="
echo ""

# Display results
for pooling_type in mean attention max cls; do
    result_file="output/ablation/pooling_${pooling_type}/results/test_results_latest.json"
    if [ -f "$result_file" ]; then
        echo "=== Pooling: $pooling_type ==="
        python3 -c "import json; d=json.load(open('$result_file')); m=d['metrics']; print(f\"Accuracy: {m['accuracy']:.4f} | Balanced: {m['balanced_accuracy']:.4f} | F1: {m['f1']:.4f} | AUC: {m['auc']:.4f}\")"
        echo ""
    fi
done
