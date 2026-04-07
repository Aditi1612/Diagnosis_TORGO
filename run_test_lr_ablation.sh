#!/bin/bash

# Fix bitsandbytes CUDA error
export BITSANDBYTES_NOWELCOME=1
export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "Testing Learning Rate Ablation Models"
echo "========================================="
echo ""

# Test LR=1e-3
echo "--- Testing LR=1e-3 ---"
if [ -f "output/wav2vec2_speaker_lr1e3/checkpoints/best_model.pth" ]; then
    python scripts/train_wav2vec2.py \
        --data_root ./TORGO_database \
        --output_dir ./output/wav2vec2_speaker_lr1e3 \
        --model_type wav2vec2 \
        --model_name facebook/wav2vec2-base \
        --split_mode speaker \
        --pooling mean \
        --batch_size 8 \
        --max_length_sec 4.0 \
        --checkpoint ./output/wav2vec2_speaker_lr1e3/checkpoints/best_model.pth \
        --mode test
    echo ""
else
    echo "No checkpoint found for LR=1e-3"
    echo ""
fi

# Test LR=1e-4
echo "--- Testing LR=1e-4 (Baseline) ---"
if [ -f "output/wav2vec2_speaker_lr1e4/checkpoints/best_model.pth" ]; then
    python scripts/train_wav2vec2.py \
        --data_root ./TORGO_database \
        --output_dir ./output/wav2vec2_speaker_lr1e4 \
        --model_type wav2vec2 \
        --model_name facebook/wav2vec2-base \
        --split_mode speaker \
        --pooling mean \
        --batch_size 8 \
        --max_length_sec 4.0 \
        --checkpoint ./output/wav2vec2_speaker_lr1e4/checkpoints/best_model.pth \
        --mode test
    echo ""
else
    echo "No checkpoint found for LR=1e-4"
    echo ""
fi

# Test LR=1e-5
echo "--- Testing LR=1e-5 ---"
if [ -f "output/wav2vec2_speaker_lr1e5/checkpoints/best_model.pth" ]; then
    python scripts/train_wav2vec2.py \
        --data_root ./TORGO_database \
        --output_dir ./output/wav2vec2_speaker_lr1e5 \
        --model_type wav2vec2 \
        --model_name facebook/wav2vec2-base \
        --split_mode speaker \
        --pooling mean \
        --batch_size 8 \
        --max_length_sec 4.0 \
        --checkpoint ./output/wav2vec2_speaker_lr1e5/checkpoints/best_model.pth \
        --mode test
    echo ""
else
    echo "No checkpoint found for LR=1e-5"
    echo ""
fi

# Test LR=1e-6
echo "--- Testing LR=1e-6 ---"
if [ -f "output/wav2vec2_speaker_lr1e6/checkpoints/best_model.pth" ]; then
    python scripts/train_wav2vec2.py \
        --data_root ./TORGO_database \
        --output_dir ./output/wav2vec2_speaker_lr1e6 \
        --model_type wav2vec2 \
        --model_name facebook/wav2vec2-base \
        --split_mode speaker \
        --pooling mean \
        --batch_size 8 \
        --max_length_sec 4.0 \
        --checkpoint ./output/wav2vec2_speaker_lr1e6/checkpoints/best_model.pth \
        --mode test
    echo ""
else
    echo "No checkpoint found for LR=1e-6"
    echo ""
fi

echo "========================================="
echo "Testing Complete!"
echo "========================================="
echo ""
echo "Check results in:"
echo "  - output/wav2vec2_speaker_lr1e3/results/"
echo "  - output/wav2vec2_speaker_lr1e4/results/"
echo "  - output/wav2vec2_speaker_lr1e5/results/"
echo "  - output/wav2vec2_speaker_lr1e6/results/"
