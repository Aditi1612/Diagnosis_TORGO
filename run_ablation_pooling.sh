#!/bin/bash

echo "========================================="
echo "Pooling Strategy Ablation Study"
echo "Wav2Vec2 Dysarthria Classifier"
echo "========================================="
echo ""
echo "Testing pooling strategies: mean, max, attention, cls"
echo "Model: Wav2Vec2-Base"
echo "Learning Rate: 1e-5 (proven optimal)"
echo ""

# Experiment 1: Mean Pooling (BASELINE - 98.97%)
echo "========================================="
echo "Experiment 1: Mean Pooling (BASELINE)"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/pooling_mean \
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
    --wandb_project "dysarthria-ablation-pooling" \
    --wandb_run_name "pooling-mean-baseline" \
    --mode train

# Experiment 2: Max Pooling
echo ""
echo "========================================="
echo "Experiment 2: Max Pooling"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/pooling_max \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling max \
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
    --wandb_project "dysarthria-ablation-pooling" \
    --wandb_run_name "pooling-max" \
    --mode train

# Experiment 3: Attention Pooling
echo ""
echo "========================================="
echo "Experiment 3: Attention Pooling"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/pooling_attention \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling attention \
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
    --wandb_project "dysarthria-ablation-pooling" \
    --wandb_run_name "pooling-attention" \
    --mode train

# Experiment 4: CLS Token Pooling
echo ""
echo "========================================="
echo "Experiment 4: CLS Token Pooling"
echo "========================================="
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/pooling_cls \
    --model_type wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling cls \
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
    --wandb_project "dysarthria-ablation-pooling" \
    --wandb_run_name "pooling-cls" \
    --mode train

echo ""
echo "========================================="
echo "Pooling Ablation Study Complete!"
echo "========================================="
echo ""
echo "Results saved in ./output/ablation/pooling_*"
echo ""
echo "Strategies tested:"
echo "  1. Mean Pooling (baseline: 98.97%)"
echo "  2. Max Pooling"
echo "  3. Attention Pooling (learnable)"
echo "  4. CLS Token Pooling"
