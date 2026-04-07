#!/bin/bash

echo "========================================="
echo "Multi-Model Learning Rate Ablation Study"
echo "========================================="
echo ""
echo "Models: wav2vec2-large, HuBERT"
echo "Learning Rates: 1e-3, 1e-4, 1e-5, 1e-6"
echo "Total experiments: 8 (2 models × 4 learning rates)"
echo ""
echo "Note: Wav2Vec2-Base already completed (98.97% @ 1e-4)"
echo ""

# ============================================
# WAV2VEC2-LARGE EXPERIMENTS
# ============================================
echo ""
echo "========================================="
echo "WAV2VEC2-LARGE MODEL"
echo "========================================="

# Wav2Vec2-Large + LR=1e-3
echo ""
echo "--- Wav2Vec2-Large: LR = 1e-3 ---"
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/wav2vec2_large_lr1e3 \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-large \
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
    --wandb_project "dysarthria-ablation" \
    --wandb_run_name "wav2vec2-large-lr1e3" \
    --mode train

# Wav2Vec2-Large + LR=1e-4
echo ""
echo "--- Wav2Vec2-Large: LR = 1e-4 ---"
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/wav2vec2_large_lr1e4 \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-large \
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
    --wandb_project "dysarthria-ablation" \
    --wandb_run_name "wav2vec2-large-lr1e4" \
    --mode train

# Wav2Vec2-Large + LR=1e-5
echo ""
echo "--- Wav2Vec2-Large: LR = 1e-5 ---"
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/wav2vec2_large_lr1e5 \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-large \
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
    --wandb_project "dysarthria-ablation" \
    --wandb_run_name "wav2vec2-large-lr1e5" \
    --mode train

# Wav2Vec2-Large + LR=1e-6
echo ""
echo "--- Wav2Vec2-Large: LR = 1e-6 ---"
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/wav2vec2_large_lr1e6 \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-large \
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
    --wandb_project "dysarthria-ablation" \
    --wandb_run_name "wav2vec2-large-lr1e6" \
    --mode train

# ============================================
# HUBERT EXPERIMENTS
# ============================================
echo ""
echo "========================================="
echo "HUBERT MODEL"
echo "========================================="

# HuBERT + LR=1e-3
echo ""
echo "--- HuBERT: LR = 1e-3 ---"
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/hubert_lr1e3 \
    --model_type hubert \
    --model_name facebook/hubert-base-ls960 \
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
    --wandb_project "dysarthria-ablation" \
    --wandb_run_name "hubert-lr1e3" \
    --mode train

# HuBERT + LR=1e-4
echo ""
echo "--- HuBERT: LR = 1e-4 ---"
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/hubert_lr1e4 \
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
    --seed 42 \
    --num_workers 4 \
    --save_interval 5 \
    --use_wandb \
    --wandb_project "dysarthria-ablation" \
    --wandb_run_name "hubert-lr1e4" \
    --mode train

# HuBERT + LR=1e-5
echo ""
echo "--- HuBERT: LR = 1e-5 ---"
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/hubert_lr1e5 \
    --model_type hubert \
    --model_name facebook/hubert-base-ls960 \
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
    --wandb_project "dysarthria-ablation" \
    --wandb_run_name "hubert-lr1e5" \
    --mode train

# HuBERT + LR=1e-6
echo ""
echo "--- HuBERT: LR = 1e-6 ---"
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/hubert_lr1e6 \
    --model_type hubert \
    --model_name facebook/hubert-base-ls960 \
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
    --wandb_project "dysarthria-ablation" \
    --wandb_run_name "hubert-lr1e6" \
    --mode train

echo ""
echo "========================================="
echo "Ablation Study Complete!"
echo "========================================="
echo ""
echo "Results saved in ./output/ablation/"
echo ""
echo "Models tested:"
echo "  1. Wav2Vec2-Large"
echo "  2. HuBERT"
echo ""
echo "Learning rates tested: 1e-3, 1e-4, 1e-5, 1e-6"
echo ""
echo "Total experiments: 8"
echo ""
echo "Baseline (already completed): Wav2Vec2-Base @ 1e-4 = 98.97%"

