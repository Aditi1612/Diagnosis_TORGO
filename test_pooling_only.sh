#!/bin/bash

source /home/work/miniconda3/bin/activate diag_paper

echo "========================================="
echo "Testing Pooling Ablation Models"
echo "========================================="
echo ""

for pooling in mean attention first; do
    if [ -d "output/ablation/pooling_$pooling/checkpoints" ]; then
        echo ""
        echo "--- Testing Pooling: $pooling ---"
        python scripts/train_wav2vec2.py \
            --data_root ./TORGO_database \
            --output_dir ./output/ablation/pooling_$pooling \
            --model_type wav2vec2 \
            --model_name facebook/wav2vec2-base \
            --split_mode speaker \
            --pooling $pooling \
            --batch_size 8 \
            --seed 42 \
            --num_workers 4 \
            --mode test \
            --checkpoint ./output/ablation/pooling_$pooling/checkpoints/best_model.pth
    else
        echo ""
        echo "--- Pooling $pooling: Not trained yet ---"
    fi
done

echo ""
echo ""
echo "========================================="
echo "POOLING RESULTS SUMMARY"
echo "========================================="
for pooling in mean attention first; do
    result_file="output/ablation/pooling_$pooling/results/test_results_latest.json"
    if [ -f "$result_file" ]; then
        acc=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['accuracy']:.4f}\")")
        bal=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['balanced_accuracy']:.4f}\")")
        f1=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['f1']:.4f}\")")
        auc=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['auc']:.4f}\")")
        echo ""
        echo "=== Pooling: $pooling ==="
        echo "Accuracy: $acc | Balanced: $bal | F1: $f1 | AUC: $auc"
    fi
done

echo ""
echo "========================================="
