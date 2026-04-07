#!/bin/bash

echo "========================================="
echo "Stage 2: Clinical Explanation Generation"
echo "========================================="
echo ""
echo "Using trained Wav2Vec2 model to generate"
echo "clinical explanations with DeepSeek-R1"
echo ""

# Default paths
CHECKPOINT=${1:-"./output/wav2vec2_speaker/checkpoints/best_model.pth"}
AUDIO=${2:-""}

if [ -z "$AUDIO" ]; then
    echo "Usage: ./run_stage2.sh [checkpoint] <audio_file>"
    echo ""
    echo "Example:"
    echo "  ./run_stage2.sh ./output/wav2vec2_speaker/checkpoints/best_model.pth ./test_audio.wav"
    echo ""
    echo "Or for batch processing:"
    echo "  python scripts/stage2_explain.py --checkpoint $CHECKPOINT --audio_dir ./TORGO_database/M02/Session1/wav_headMic"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo "Audio: $AUDIO"
echo ""

python scripts/stage2_explain.py \
    --checkpoint "$CHECKPOINT" \
    --audio "$AUDIO" \
    --explanation_level clinical \
    --output ./output/explanation_result.json
