#!/bin/bash

# Fix bitsandbytes CUDA error
export BITSANDBYTES_NOWELCOME=1
export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "Learning Rate Ablation Study"
echo "Wav2Vec2-Base ONLY (other models collapse)"
echo "========================================="
echo ""
echo "Model: Wav2Vec2-Base"
echo "Testing learning rates: 1e-3, 1e-4 (baseline), 1e-5, 1e-6"
echo "Note: Other models excluded due to class collapse"
echo ""

# Experiment 1: LR = 1e-3
echo "========================================="
echo "Experiment 1: Learning Rate = 1e-3"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/wav2vec2_speaker_lr1e3 \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --batch_size 8 \
    --num_epochs 30 \
    --learning_rate 1e-3 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --max_length_sec 4.0 \
    --seed 42 \
    --num_workers 4 \
    --save_interval 5 \
    --use_wandb \
    --wandb_project "dysarthria-wav2vec2-ablation" \
    --wandb_run_name "lr-1e-3" \
    --mode train

# Experiment 2: LR = 1e-4 (Original best - 98.97% accuracy)
echo ""
echo "========================================="
echo "Experiment 2: Learning Rate = 1e-4 (BASELINE)"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/wav2vec2_speaker_lr1e4 \
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
    --seed 42 \
    --num_workers 4 \
    --save_interval 5 \
    --use_wandb \
    --wandb_project "dysarthria-wav2vec2-ablation" \
    --wandb_run_name "lr-1e-4-baseline" \
    --mode train

# Experiment 3: LR = 1e-5
echo ""
echo "========================================="
echo "Experiment 3: Learning Rate = 1e-5"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/wav2vec2_speaker_lr1e5 \
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
    --wandb_project "dysarthria-wav2vec2-ablation" \
    --wandb_run_name "lr-1e-5" \
    --mode train

# Experiment 4: LR = 1e-6
echo ""
echo "========================================="
echo "Experiment 4: Learning Rate = 1e-6"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/wav2vec2_speaker_lr1e6 \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --batch_size 8 \
    --num_epochs 30 \
    --learning_rate 1e-6 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --max_length_sec 4.0 \
    --seed 42 \
    --num_workers 4 \
    --save_interval 5 \
    --use_wandb \
    --wandb_project "dysarthria-wav2vec2-ablation" \
    --wandb_run_name "lr-1e-6" \
    --mode train

echo ""
echo "========================================="
echo "Ablation Study Complete!"
echo "========================================="
echo "Results saved in:"
echo "  - ./output/wav2vec2_speaker_lr1e3"
echo "  - ./output/wav2vec2_speaker_lr1e4"
echo "  - ./output/wav2vec2_speaker_lr1e5"
echo "  - ./output/wav2vec2_speaker_lr1e6"
