#!/bin/bash

echo "========================================="
echo "Audio Length Ablation Study"
echo "Wav2Vec2 Dysarthria Classifier"
echo "========================================="
echo ""
echo "Testing audio lengths: 2.0s, 3.0s, 4.0s, 5.0s, 6.0s"
echo "Model: Wav2Vec2-Base"
echo "Learning Rate: 1e-5 (proven optimal)"
echo "Pooling: mean"
echo ""

# Experiment 1: 2.0 seconds
echo "========================================="
echo "Experiment 1: Max Length = 2.0 seconds"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/length_2sec \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --batch_size 8 \
    --num_epochs 30 \
    --learning_rate 1e-5 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --max_length_sec 2.0 \
    --seed 42 \
    --num_workers 4 \
    --save_interval 5 \
    --use_wandb \
    --wandb_project "dysarthria-ablation-length" \
    --wandb_run_name "length-2sec" \
    --mode train

# Experiment 2: 3.0 seconds
echo ""
echo "========================================="
echo "Experiment 2: Max Length = 3.0 seconds"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/length_3sec \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --batch_size 8 \
    --num_epochs 30 \
    --learning_rate 1e-5 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --max_length_sec 3.0 \
    --seed 42 \
    --num_workers 4 \
    --save_interval 5 \
    --use_wandb \
    --wandb_project "dysarthria-ablation-length" \
    --wandb_run_name "length-3sec" \
    --mode train

# Experiment 3: 4.0 seconds (BASELINE - 98.97%)
echo ""
echo "========================================="
echo "Experiment 3: Max Length = 4.0 seconds (BASELINE)"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/length_4sec \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --batch_size 8 \
    --num_epochs 30 \
    --learning_rate 1e-5 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --max_length_sec 4.0 \
    --seed 42 \
    --num_workers 4 \
    --save_interval 5 \
    --use_wandb \
    --wandb_project "dysarthria-ablation-length" \
    --wandb_run_name "length-4sec-baseline" \
    --mode train

# Experiment 4: 5.0 seconds
echo ""
echo "========================================="
echo "Experiment 4: Max Length = 5.0 seconds"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/length_5sec \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --batch_size 8 \
    --num_epochs 30 \
    --learning_rate 1e-5 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --max_length_sec 5.0 \
    --seed 42 \
    --num_workers 4 \
    --save_interval 5 \
    --use_wandb \
    --wandb_project "dysarthria-ablation-length" \
    --wandb_run_name "length-5sec" \
    --mode train

# Experiment 5: 6.0 seconds
echo ""
echo "========================================="
echo "Experiment 5: Max Length = 6.0 seconds"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/length_6sec \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --batch_size 8 \
    --num_epochs 30 \
    --learning_rate 1e-5 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --max_length_sec 6.0 \
    --seed 42 \
    --num_workers 4 \
    --save_interval 5 \
    --use_wandb \
    --wandb_project "dysarthria-ablation-length" \
    --wandb_run_name "length-6sec" \
    --mode train

echo ""
echo "========================================="
echo "Audio Length Ablation Study Complete!"
echo "========================================="
echo ""
echo "Results saved in ./output/ablation/length_*"
echo ""
echo "Audio lengths tested:"
echo "  1. 2.0 seconds (shorter, less context)"
echo "  2. 3.0 seconds"
echo "  3. 4.0 seconds (baseline: 98.97%)"
echo "  4. 5.0 seconds"
echo "  5. 6.0 seconds (longer, more context)"
echo ""
echo "Analysis: Check if longer audio captures more dysarthric patterns"
