#!/bin/bash

echo "========================================="
echo "Testing Wav2Vec2 Best Model"
echo "========================================="

CHECKPOINT=${1:-"./output/wav2vec2_speaker_hubert/checkpoints/best_model.pth"}
echo "Checkpoint: $CHECKPOINT"

python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/wav2vec2_speaker_hubert \
    --model_type hubert \
    --model_name facebook/hubert-base-ls960 \
    --split_mode speaker \
    --pooling mean \
    --batch_size 8 \
    --max_length_sec 4.0 \
    --mode test \
    --checkpoint "$CHECKPOINT"
