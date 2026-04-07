#!/bin/bash

echo "========================================="
echo "Wav2Vec2 Dysarthria Classifier"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Model: wav2vec2-base (UNFROZEN encoder)"
echo "  - Split: Speaker-independent (severity-balanced)"
echo "  - Train: 5 dysarthric + 4 control speakers"
echo "  - Val:   2 dysarthric + 2 control speakers"
echo "  - Test:  1 dysarthric + 1 control speakers"
echo "  - LR: 1e-5 (for fine-tuning full model)"
echo ""

python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/wav2vec2_speaker_hubert \
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
    --use_wandb \
    --wandb_project "dysarthria-hubert" \
    --wandb_run_name "hubert-unfrozen-speaker" \
    --mode train
