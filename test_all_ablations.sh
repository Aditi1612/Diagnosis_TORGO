#!/bin/bash

source /home/work/miniconda3/bin/activate diag_paper

echo "========================================="
echo "Testing All Ablation Models"
echo "========================================="
echo ""

# Test Pooling Strategies
echo "========================================="
echo "POOLING ABLATION RESULTS"
echo "========================================="

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

# Test Audio Length Experiments
echo ""
echo ""
echo "========================================="
echo "AUDIO LENGTH ABLATION RESULTS"
echo "========================================="

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
echo "SUMMARY OF ALL RESULTS"
echo "========================================="
echo ""
echo "=== POOLING STRATEGIES ==="
for pooling in mean attention first; do
    result_file="output/ablation/pooling_$pooling/results/test_results_latest.json"
    if [ -f "$result_file" ]; then
        acc=$(python -c "import json; print(f\"{json.load(open('$result_file'))['accuracy']:.4f}\")")
        bal=$(python -c "import json; print(f\"{json.load(open('$result_file'))['balanced_accuracy']:.4f}\")")
        f1=$(python -c "import json; print(f\"{json.load(open('$result_file'))['f1_score']:.4f}\")")
        auc=$(python -c "import json; print(f\"{json.load(open('$result_file'))['auc']:.4f}\")")
        echo "Pooling: $pooling | Accuracy: $acc | Balanced: $bal | F1: $f1 | AUC: $auc"
    fi
done

echo ""
echo "=== AUDIO LENGTHS ==="
echo "Baseline (4.0s): 99.08% (from wav2vec2_speaker_lr1e5)"
for length in 2sec 3sec 5sec 6sec; do
    result_file="output/ablation/length_$length/results/test_results_latest.json"
    if [ -f "$result_file" ]; then
        acc=$(python -c "import json; print(f\"{json.load(open('$result_file'))['accuracy']:.4f}\")")
        bal=$(python -c "import json; print(f\"{json.load(open('$result_file'))['balanced_accuracy']:.4f}\")")
        f1=$(python -c "import json; print(f\"{json.load(open('$result_file'))['f1_score']:.4f}\")")
        auc=$(python -c "import json; print(f\"{json.load(open('$result_file'))['auc']:.4f}\")")
        echo "Length: $length | Accuracy: $acc | Balanced: $bal | F1: $f1 | AUC: $auc"
    fi
done

echo ""
echo "========================================="
echo "Testing Complete!"
echo "========================================="
