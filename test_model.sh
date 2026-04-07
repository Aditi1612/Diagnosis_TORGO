#!/bin/bash

# Test DeepThink-Speech Model on TORGO Test Set

echo "========================================="
echo "DeepThink-Speech: Model Testing"
echo "========================================="
echo ""

# Set default paths
MODEL_DIR="./output/deepthink_fixed_v2"
CHECKPOINT="${MODEL_DIR}/checkpoints/best_model.pth"
DATA_ROOT="./TORGO_database"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    find ./output -name "best_model.pth" -type f
    echo ""
    echo "Usage: $0 [checkpoint_path]"
    echo "Example: $0 ./output/deepthink_fixed_v2/checkpoints/best_model.pth"
    exit 1
fi

# Allow override via command line
if [ ! -z "$1" ]; then
    CHECKPOINT="$1"
    echo "Using checkpoint: $CHECKPOINT"
fi

echo "Testing model..."
echo "  Checkpoint: $CHECKPOINT"
echo "  Data: $DATA_ROOT"
echo ""

python scripts/train_deepthink_speech.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$MODEL_DIR" \
    --mode test \
    --checkpoint "$CHECKPOINT" \
    --n_mfcc 40 \
    --hidden_dim 128 \
    --num_layers 2 \
    --num_heads 4 \
    --batch_size 64

echo ""
echo "========================================="
echo "Testing completed!"
echo "Results saved to: ${MODEL_DIR}/test_results.json"
echo "========================================="
