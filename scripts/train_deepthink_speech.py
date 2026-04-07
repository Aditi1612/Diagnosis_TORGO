"""
Training and Inference Script for DeepThink-Speech
Integrates BiLSTM-MHA model with explainable AI framework
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import librosa
import importlib.util
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import DeepThinkSpeech directly without going through models/__init__.py
# This avoids dependency conflicts with DepMamba/speechbrain
deepthink_path = Path(__file__).parent.parent / "models" / "DeepThinkSpeech.py"
spec = importlib.util.spec_from_file_location("DeepThinkSpeech", deepthink_path)
deepthink_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepthink_module)
BiLSTM_MHA_Dysarthria = deepthink_module.BiLSTM_MHA_Dysarthria
HybridExplainableModel = deepthink_module.HybridExplainableModel

from datasets_process.torgo import create_torgo_dataloaders
from utils.acoustic_features import AcousticFeatureExtractor
from utils.llm_explainer import ClinicalExplainer


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DeepThinkSpeechTrainer:
    """Trainer for DeepThink-Speech model"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Print device information
        print("\n" + "="*60)
        print("DEVICE INFORMATION")
        print("="*60)
        if torch.cuda.is_available():
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ Using device: {self.device}")
        else:
            print(f"✗ GPU NOT available - using CPU")
            print(f"✗ Training will be VERY slow on CPU")
        print("="*60 + "\n")

        # Create output directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)

        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Initialize wandb if enabled
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                entity=args.wandb_entity
            )

        # Load data
        print("Loading TORGO dataset...")
        self.train_loader, self.val_loader, self.test_loader = create_torgo_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            n_mfcc=args.n_mfcc,  # Data loader automatically adds delta and delta-delta
            num_workers=args.num_workers,
            seed=args.seed
        )

        # Initialize model
        print("Initializing model...")
        self.model = BiLSTM_MHA_Dysarthria(
            input_dim=args.n_mfcc * 3,  # MFCC + delta + delta-delta
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_classes=2
        ).to(self.device)

        # Initialize optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Use Cosine Annealing for predictable LR schedule
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-6
        )

        # MILD class weights - balanced but not extreme
        class_weights = self._compute_class_weights()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print(f"Using CrossEntropyLoss with MILD class weights (max ratio 1.5x)")

        # Initialize explainability components
        self.acoustic_extractor = AcousticFeatureExtractor(sample_rate=16000)
        self.clinical_explainer = ClinicalExplainer()

        # Training state
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.patience_counter = 0
        self.early_stop_patience = None  # Disabled - train for all epochs

    def _compute_class_weights(self):
        """Compute class weights from training data to handle class imbalance"""
        all_labels = []
        for _, labels, _ in self.train_loader:
            all_labels.extend(labels.numpy())

        all_labels = np.array(all_labels)
        class_counts = np.bincount(all_labels)

        print(f"\nClass distribution in training set:")
        print(f"  Class 0 (Control): {class_counts[0]} samples")
        print(f"  Class 1 (Dysarthric): {class_counts[1]} samples")

        # Compute inverse frequency weights
        total_samples = len(all_labels)
        class_weights = total_samples / (len(class_counts) * class_counts)

        print(f"\nComputed class weights:")
        print(f"  Class 0 (Control) weight: {class_weights[0]:.4f}")
        print(f"  Class 1 (Dysarthric) weight: {class_weights[1]:.4f}")

        # MILD class weights - clip to 1.5x ratio max
        # Strong weights cause collapse, no weights also causes collapse
        max_weight_ratio = 1.5
        weight_ratio = max(class_weights) / min(class_weights)
        if weight_ratio > max_weight_ratio:
            print(f"  Warning: Weight ratio {weight_ratio:.2f} too extreme, clipping to {max_weight_ratio}")
            if class_weights[0] > class_weights[1]:
                class_weights[0] = class_weights[1] * max_weight_ratio
            else:
                class_weights[1] = class_weights[0] * max_weight_ratio
            print(f"  Clipped weights - Control: {class_weights[0]:.4f}, Dysarthric: {class_weights[1]:.4f}")

        return torch.FloatTensor(class_weights).to(self.device)

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} [Train]")

        for batch_idx, (mfcc, labels, metadata) in enumerate(pbar):
            mfcc = mfcc.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Small feature noise to help generalization (not too much)
            noise = torch.randn_like(mfcc) * 0.05  # 5% noise
            mfcc = mfcc + noise

            logits = self.model(mfcc)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Log to tensorboard
            if self.global_step % self.args.log_interval == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)

            self.global_step += 1

        # Epoch statistics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

        # Per-class metrics for training too
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        ctrl_mask = all_labels_np == 0
        dys_mask = all_labels_np == 1
        train_ctrl_acc = (all_preds_np[ctrl_mask] == 0).mean() if ctrl_mask.sum() > 0 else 0
        train_dys_acc = (all_preds_np[dys_mask] == 1).mean() if dys_mask.sum() > 0 else 0
        pred_ctrl = (all_preds_np == 0).sum()
        pred_dys = (all_preds_np == 1).sum()

        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
              f"Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
        print(f"  >> [TRAIN] Control acc={train_ctrl_acc:.2%}, Dysarthric acc={train_dys_acc:.2%}")
        print(f"  >> [TRAIN] Predictions: {pred_ctrl} control, {pred_dys} dysarthric")

        return avg_loss, accuracy, precision, recall, f1

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} [Val]")

            for mfcc, labels, metadata in pbar:
                mfcc = mfcc.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(mfcc)
                loss = self.criterion(logits, labels)

                # Statistics
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of dysarthric class

        # Epoch statistics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        auc = roc_auc_score(all_labels, all_probs)

        # Calculate per-class metrics to detect collapse
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        # Class-wise accuracy
        control_mask = all_labels_np == 0
        dysarthric_mask = all_labels_np == 1
        control_acc = (all_preds_np[control_mask] == 0).mean() if control_mask.sum() > 0 else 0
        dysarthric_acc = (all_preds_np[dysarthric_mask] == 1).mean() if dysarthric_mask.sum() > 0 else 0

        # Prediction distribution
        pred_control = (all_preds_np == 0).sum()
        pred_dysarthric = (all_preds_np == 1).sum()

        print(f"Epoch {epoch+1} - Val Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
              f"Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"  >> [VAL] Control acc={control_acc:.2%}, Dysarthric acc={dysarthric_acc:.2%}")
        print(f"  >> [VAL] Predictions: {pred_control} control, {pred_dysarthric} dysarthric")

        # Warning if model is collapsing
        if control_acc < 0.2 or dysarthric_acc < 0.2:
            print(f"  ⚠️ WARNING: Model collapsing! One class < 20% accuracy")

        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Val/F1', f1, epoch)
        self.writer.add_scalar('Val/AUC', auc, epoch)
        self.writer.add_scalar('Val/Control_Acc', control_acc, epoch)
        self.writer.add_scalar('Val/Dysarthric_Acc', dysarthric_acc, epoch)

        return avg_loss, accuracy, precision, recall, f1, auc, control_acc, dysarthric_acc

    def train(self):
        """Full training loop"""
        print(f"\nStarting training on {self.device}...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        if torch.cuda.is_available():
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print()

        for epoch in range(self.args.num_epochs):
            # Train
            train_loss, train_acc, train_prec, train_rec, train_f1 = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, ctrl_acc, dys_acc = self.validate(epoch)

            # Calculate overfitting gap and class balance
            train_val_gap = train_acc - val_acc
            class_balance = min(ctrl_acc, dys_acc) / max(ctrl_acc, dys_acc) if max(ctrl_acc, dys_acc) > 0 else 0
            print(f"  >>> Overfitting gap: {train_val_gap:.4f} | Class balance ratio: {class_balance:.2f}")

            # Log to wandb if enabled (consolidated logging per epoch)
            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'train/precision': train_prec,
                    'train/recall': train_rec,
                    'train/f1': train_f1,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'val/precision': val_prec,
                    'val/recall': val_rec,
                    'val/f1': val_f1,
                    'val/auc': val_auc,
                    'val/control_acc': ctrl_acc,
                    'val/dysarthric_acc': dys_acc,
                    'val/class_balance': class_balance,
                    'overfitting_gap': train_val_gap,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch, commit=True)

            # Learning rate scheduling (Cosine Annealing uses epoch, not loss)
            self.scheduler.step(epoch)

            # Track best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint based on validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"  >>> New best model saved! Val Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)

        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")
        self.writer.close()

        # Close wandb if enabled
        if self.args.use_wandb:
            wandb.finish()

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'args': vars(self.args)
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'

        torch.save(checkpoint, path)

    def test(self, checkpoint_path=None):
        """Test the model and generate explanations"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        sample_explanations = []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")

            for batch_idx, (mfcc, labels, metadata) in enumerate(pbar):
                mfcc = mfcc.to(self.device)
                labels = labels.to(self.device)

                # Forward pass with attention
                logits, attention_weights = self.model(mfcc, return_attention=True)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

                # Generate explanation for first sample of first batch
                if batch_idx == 0 and len(sample_explanations) < 3:
                    for i in range(min(3, mfcc.shape[0])):
                        explanation = self.generate_explanation(
                            mfcc[i].cpu().numpy(),
                            preds[i].item(),
                            probs[i, 1].item(),
                            attention_weights[i].cpu().numpy(),
                            metadata['audio_path'][i]
                        )
                        sample_explanations.append(explanation)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_preds)

        print("\n=== Test Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)

        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'sample_explanations': sample_explanations
        }

        results_path = self.output_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_path}")

        # Print sample explanations
        print("\n=== Sample Explanations ===")
        for i, exp in enumerate(sample_explanations):
            print(f"\n--- Sample {i+1} ---")
            print(exp['explanation'])

        return results

    def generate_explanation(self, mfcc, prediction, confidence, attention_weights, audio_path):
        """Generate full explanation for a single sample"""
        # Load original audio for acoustic feature extraction
        try:
            audio, sr = librosa.load(audio_path, sr=16000)

            # Get high attention regions
            attention_scores = attention_weights.mean(axis=0).mean(axis=0)
            threshold = np.percentile(attention_scores, 75)
            high_attention_mask = attention_scores >= threshold

            # Extract acoustic features from high-attention regions
            acoustic_features = self.acoustic_extractor.extract_all_features(
                audio,
                high_attention_mask=high_attention_mask
            )

            # Generate clinical explanation
            explanation = self.clinical_explainer.generate_explanation(
                prediction=prediction,
                confidence=confidence,
                attention_weights=attention_weights,
                acoustic_features=acoustic_features,
                audio_metadata={'audio_path': audio_path},
                explanation_level="clinical"
            )

            return explanation

        except Exception as e:
            print(f"Error generating explanation: {e}")
            return {
                'prediction': 'Dysarthric' if prediction == 1 else 'Healthy',
                'confidence': confidence,
                'explanation': f"Error generating explanation: {e}"
            }


def main():
    parser = argparse.ArgumentParser(description='Train DeepThink-Speech Model')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='/home/work/Aditi/diag_paper/TORGO_database',
                       help='Root directory of TORGO database')
    parser.add_argument('--output_dir', type=str, default='./output/deepthink_speech',
                       help='Output directory for checkpoints and logs')

    # Model arguments
    parser.add_argument('--n_mfcc', type=int, default=40,
                       help='Number of MFCC coefficients')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of BiLSTM layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval (steps)')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Checkpoint save interval (epochs)')

    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='deepthink-speech',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity (username or team)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name (auto-generated if not specified)')

    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'],
                       help='Mode: train, test, or both')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for testing')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize trainer
    trainer = DeepThinkSpeechTrainer(args)

    # Run
    if args.mode in ['train', 'both']:
        trainer.train()

    if args.mode in ['test', 'both']:
        trainer.test(checkpoint_path=args.checkpoint)


if __name__ == '__main__':
    main()
