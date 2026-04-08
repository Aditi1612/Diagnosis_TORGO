**Automated Dysarthria Detection With Clinician-Interpretable Explanations: Validation of a Self-Supervised Speech Model With Speech-Language Pathologists**

A two-stage framework combining a fine-tuned Wav2Vec2 classifier with LLM-based chain-of-thought reasoning to produce structured clinical explanations for dysarthric speech alongside diagnostic predictions.

---

## Results (TORGO, Speaker-Independent)

| Metric | Value |
|---|---|
| Accuracy | **99.66%** |
| Balanced Accuracy | **99.68%** |
| F1 Score | **99.63%** |
| AUC | **99.99%** |

SLP validation (N=3, 20 cases): overall clinical usefulness **4.63/5.0**, Fleiss' κ = 0.82, 96.7% diagnostic concordance.

---

## Environment

```
OS          : Ubuntu 24.04.2 LTS
GPU         : NVIDIA H100 (80GB)
Framework   : PyTorch 2.0+, CUDA 12.1
Python      : 3.10+
```

---

## Installation

```bash
git clone <repo>
cd Deep-think
pip install -r requirements.txt
```

**Montreal Forced Aligner** (optional — for phoneme-level analysis):
```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

---

## Dataset

Download the [TORGO corpus](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) and place it at `./TORGO_database/`.

```
TORGO_database/
├── F01/, F03/, F04/              # Dysarthric female speakers
├── M01/, M02/, M03/, M04/, M05/ # Dysarthric male speakers
├── FC01/, FC02/, FC03/           # Control female speakers
└── MC01/, MC02/, MC03/, MC04/   # Control male speakers
```

- 15 speakers (8 dysarthric, 7 control)
- 8,216 utterances after preprocessing (resampled to 16 kHz, capped at 4.0 s)
- Speaker-independent split: train 9 speakers / val 4 / test 2

---

## Repository Structure

```
├── models/
│   ├── Wav2Vec2Classifier.py        # Stage 1: Wav2Vec2 fine-tuning + attention extraction
│   ├── Wav2Vec2ClassifierAblation.py   # Ablation variant
│   ├── DeepThinkSpeech.py           # Full two-stage pipeline
│   ├── base.py
│   └── __init__.py
├── datasets_process/
│   ├── torgo.py                     # TORGO dataset loader
│   └── torgo_raw.py                 # Raw waveform loader
├── train_eval/
│   ├── train_val.py                 # Training and validation loop
│   ├── losses.py                    # Loss functions
│   └── utils.py                     # Training utilities
├── utils/
│   ├── acoustic_features.py         # 52-feature extraction (Praat + librosa)
│   ├── llm_explainer.py             # LLM prompt + explanation generation
│   └── forced_aligner.py            # MFA phoneme alignment (optional)
├── scripts/
│   ├── train_wav2vec2.py            # Main training script (Stage 1)
│   ├── train_deepthink_speech.py    # End-to-end pipeline training
│   ├── train_ablation.py            # Extended ablation training
│   ├── stage2_explain.py            # Stage 2 explanation inference
│   └── compute_results.py           # Results aggregation
│
├── run_stage2_explanation.sh        # Run Stage 2 on best checkpoint
├── run_stage2.sh                    # Full Stage 2 pipeline
├── run_stage2_test.py               # Stage 2 test script
│
├── run_ablation_lr.sh               # Learning rate ablation (train)
├── run_ablation_pooling.sh          # Pooling strategy ablation (train)
├── run_ablation_audio_length.sh     # Segment length ablation (train)
├── run_ablation_base_model.sh       # Base model comparison (train)
├── run_ablation_models_lr_fixed.sh  # Multi-model LR ablation (train)
│
├── run_test_lr_ablation.sh          # Learning rate ablation (evaluate)
├── test_all_ablations.sh            # All ablations (evaluate)
├── test_pooling_only.sh             # Pooling ablation (evaluate)
├── test_audio_length.sh             # Segment length ablation (evaluate)
│
├── evaluate_explanations.py         # Automatic explanation quality metrics
└── verify_stage2_metrics.py         # Verify Stage 2 output metrics
```

---

## Stage 1: Training the Classifier

```bash
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/wav2vec2_speaker \
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
    --mode train
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `facebook/wav2vec2-base` | HuggingFace model ID |
| `--model_type` | `wav2vec2` | `wav2vec2` or `hubert` |
| `--split_mode` | `speaker` | `speaker` (independent) or `sample` |
| `--pooling` | `mean` | `mean`, `attention`, or `first` |
| `--learning_rate` | `1e-4` | Use `1e-5` for best results |
| `--max_length_sec` | `4.0` | Audio segment length in seconds |
| `--mode` | `train` | `train`, `test`, or `both` |

### Testing a checkpoint

```bash
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/wav2vec2_speaker \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --checkpoint ./output/wav2vec2_speaker/checkpoints/best_model.pth \
    --mode test
```

---

## Stage 2: Clinical Explanation Generation

Runs LLM-based explanation generation on the best Stage 1 checkpoint:

```bash
bash run_stage2_explanation.sh
```

Or directly:

```bash
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
```

Outputs are saved to `output/stage2_explanations/` as per-sample JSON explanations and aggregate metrics.

---

## Ablation Studies

### Learning Rate Sensitivity

```bash
bash run_ablation_lr.sh        # train
bash run_test_lr_ablation.sh   # evaluate
```

| Learning Rate | Accuracy | Bal. Acc. | Notes |
|---|---|---|---|
| 1e-3 | 53.15% | 50.00% | Class collapse |
| 1e-4 | 53.15% | 50.00% | Class collapse |
| **1e-5** | **99.66%** | **99.68%** | **Optimal** |
| 1e-6 | 93.70% | 93.28% | Underfitting |

### Temporal Pooling Strategy

```bash
bash run_ablation_pooling.sh   # train
bash test_pooling_only.sh      # evaluate
```

| Strategy | Accuracy | Bal. Acc. |
|---|---|---|
| First Token | 94.96% | 94.94% |
| Attention | 98.63% | 98.61% |
| **Mean** | **99.66%** | **99.68%** |

### Audio Segment Length

```bash
bash run_ablation_audio_length.sh   # train
bash test_audio_length.sh           # evaluate
```

| Length | Accuracy | Bal. Acc. | Errors |
|---|---|---|---|
| 2.0 s | 97.37% | 97.49% | 23 |
| 3.0 s | 96.56% | 96.68% | 30 |
| **4.0 s** | **99.66%** | **99.68%** | **3** |
| 5.0 s | 98.97% | 98.91% | 9 |
| 6.0 s | 93.36% | 92.98% | 58 |

### Base Model Comparison

```bash
bash run_ablation_base_model.sh         # Wav2Vec2-Base/Large, HuBERT
bash run_ablation_models_lr_fixed.sh    # Wav2Vec2-Large + HuBERT × LR grid
bash test_all_ablations.sh              # evaluate all
```

---

## Acoustic Features (Stage 2)

52 features extracted from high-attention regions:

| Domain | Features |
|---|---|
| Articulatory (16) | F1, F2, F3 formants, transition velocities, vowel space area |
| Phonatory (12) | Jitter, shimmer, HNR, F0 statistics |
| Temporal (8) | Speaking rate, articulation rate, pause duration/frequency |
| Spectral (16) | Spectral centroid/tilt/entropy, ZCR, MFCCs |

---

## Inference / Testing

To run the model on the test set using a trained checkpoint:

```bash
python scripts/train_wav2vec2.py \
    --data_root ./TORGO_database \
    --output_dir ./output/ablation/pooling_mean \
    --model_name facebook/wav2vec2-base \
    --split_mode speaker \
    --pooling mean \
    --checkpoint ./output/ablation/pooling_mean/checkpoints/best_model.pth \
    --mode test
```

Results are saved to `output/ablation/pooling_mean/results/`.

---

## Citation

```bibtex
@article{deepthink_speech_2026,
  title   = {Automated Dysarthria Detection With Clinician-Interpretable Explanations:
             Validation of a Self-Supervised Speech Model With Speech-Language Pathologists},
  author  = {Aditi and Ko, Myoung-Hwan and Shin, Yong-Il and Bok, Soo Kyung},
  journal = {Journal of Speech, Language, and Hearing Research},
  year    = {2026}
}

@article{rudzicz2012torgo,
  title   = {The TORGO database of acoustic and articulatory speech from speakers with dysarthria},
  author  = {Rudzicz, Frank and Namasivayam, Aravind K and Wolff, Talya},
  journal = {Language Resources and Evaluation},
  volume  = {46},
  number  = {4},
  pages   = {523--541},
  year    = {2012}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
