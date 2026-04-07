#!/bin/bash

# CONFIGURATION v10 - BALANCED APPROACH
# Problem: Model oscillates between predicting all-control or all-dysarthric
# Solution: MILD class weights (1.5x max) + moderate regularization

echo "========================================="
echo "DeepThink-Speech: v10 - Balanced Training"
echo "========================================="
echo ""
echo "CONFIGURATION v10:"
echo "  1. MILD class weights (1.5x max ratio)"
echo "  2. Small feature noise: 5%"
echo "  3. Moderate dropout: 0.5"
echo "  4. Moderate model: 128 hidden, 2 layers"
echo "  5. Moderate LR: 1e-4"
echo "  6. Weight decay: 1e-4"
echo ""
echo "Goal: Balanced predictions for both classes"
echo ""

python scripts/train_deepthink_speech.py \
    --data_root ./TORGO_database \
    --output_dir ./output/deepthink_v10_balanced \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --n_mfcc 40 \
    --hidden_dim 128 \
    --num_layers 2 \
    --num_heads 4 \
    --dropout 0.5 \
    --weight_decay 1e-4 \
    --mode train \
    --use_wandb \
    --wandb_project "deepthink-speech-dysarthria" \
    --wandb_run_name "torgo-v10-balanced"
