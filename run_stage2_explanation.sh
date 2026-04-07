#!/bin/bash

source /home/work/miniconda3/bin/activate diag_paper

echo "========================================="
echo "Stage 2: Clinical Explanation Generation"
echo "Using Best Model: pooling_mean (99.66%)"
echo "========================================="
echo ""

export BITSANDBYTES_NOWELCOME=1
export CUDA_VISIBLE_DEVICES=0

# Run Stage 2 on test set with best model
# With LLM for clinical explanations
python run_stage2_test.py \
    --checkpoint ./output/ablation/pooling_mean/checkpoints/best_model.pth \
    --data_root ./TORGO_database \
    --output_dir ./output/stage2_explanations \
    --model_name facebook/wav2vec2-base \
    --llm_model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --split_mode speaker \
    --explanation_level clinical \
    --num_samples 20 \
    --use_llm

echo ""
echo "========================================="
echo "Stage 2 Complete!"
echo "Clinical explanations saved to:"
echo "  output/stage2_explanations/"
echo "========================================="
