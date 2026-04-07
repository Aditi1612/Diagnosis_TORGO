#!/bin/bash

source /home/work/miniconda3/bin/activate diag_paper

echo "========================================="
echo "Testing Audio Length Ablation Models"
echo "========================================="
echo ""

for length in 2sec 3sec 5sec 6sec; do
    if [ -d "output/ablation/length_$length/checkpoints" ]; then
        echo ""
        echo "--- Testing Length: $length ---"
        
        # Extract numeric length
        sec=$(echo $length | sed 's/sec//')
        
        python scripts/train_wav2vec2.py \
            --data_root ./TORGO_database \
            --output_dir ./output/ablation/length_$length \
            --model_type wav2vec2 \
            --model_name facebook/wav2vec2-base \
            --split_mode speaker \
            --pooling mean \
            --batch_size 8 \
            --max_length_sec $sec.0 \
            --seed 42 \
            --num_workers 4 \
            --mode test \
            --checkpoint ./output/ablation/length_$length/checkpoints/best_model.pth
    else
        echo ""
        echo "--- Length $length: Not trained yet ---"
    fi
done

echo ""
echo ""
echo "========================================="
echo "AUDIO LENGTH RESULTS SUMMARY"
echo "========================================="
echo ""
echo "Baseline (4.0s): 99.66% (from pooling_mean)"
for length in 2sec 3sec 5sec 6sec; do
    result_file="output/ablation/length_$length/results/test_results_latest.json"
    if [ -f "$result_file" ]; then
        acc=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['accuracy']:.4f}\")")
        bal=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['balanced_accuracy']:.4f}\")")
        f1=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['f1']:.4f}\")")
        auc=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['auc']:.4f}\")")
        echo ""
        echo "=== Length: $length ==="
        echo "Accuracy: $acc | Balanced: $bal | F1: $f1 | AUC: $auc"
    fi
done

echo ""
echo "========================================="
