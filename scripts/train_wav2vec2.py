"""
Training Script for Wav2Vec2-based Dysarthria Detection
Uses pretrained Wav2Vec2 for speaker-invariant representations
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from pathlib import Path
import importlib.util
import wandb
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import directly to avoid models/__init__.py (has speechbrain conflicts)
wav2vec_path = Path(__file__).parent.parent / "models" / "Wav2Vec2Classifier.py"
spec = importlib.util.spec_from_file_location("Wav2Vec2Classifier", wav2vec_path)
wav2vec_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wav2vec_module)
Wav2Vec2DysarthriaClassifier = wav2vec_module.Wav2Vec2DysarthriaClassifier
HuBERTDysarthriaClassifier = wav2vec_module.HuBERTDysarthriaClassifier

from datasets_process.torgo_raw import create_torgo_raw_dataloaders


class Wav2Vec2Trainer:
    """Trainer for Wav2Vec2-based dysarthria detection"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "=" * 60)
        print("WAV2VEC2 DYSARTHRIA CLASSIFIER")
        print("=" * 60)
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("WARNING: No GPU available, training will be slow!")
        print("=" * 60 + "\n")

        # Output directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))

        # Wandb
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )

        # Load data
        print("Loading TORGO dataset (raw audio)...")
        self.train_loader, self.val_loader, self.test_loader = create_torgo_raw_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            max_length_sec=args.max_length_sec,
            num_workers=args.num_workers,
            seed=args.seed,
            split_mode=args.split_mode  # 'sample' for mixed speakers, 'speaker' for speaker-independent
        )

        # Initialize model
        print("\nInitializing Wav2Vec2 model...")
        if args.model_type == "hubert":
            self.model = HuBERTDysarthriaClassifier(
                num_classes=2,
                dropout=args.dropout,
                freeze_encoder=args.freeze_encoder,
                freeze_feature_extractor=True,
                pooling=args.pooling
            )
        else:
            self.model = Wav2Vec2DysarthriaClassifier(
                model_name=args.model_name,
                num_classes=2,
                dropout=args.dropout,
                freeze_encoder=args.freeze_encoder,
                freeze_feature_extractor=True,
                pooling=args.pooling
            )
        self.model = self.model.to(self.device)

        # Optimizer - only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs,
            eta_min=1e-6
        )

        # Loss - with mild class weights
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Training state
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.global_step = 0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} [Train]")

        for batch_idx, (audio, labels, metadata) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(audio)
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

            pbar.set_postfix({'loss': loss.item()})
            self.global_step += 1

        # Epoch statistics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

        # Per-class accuracy
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        ctrl_acc = (all_preds_np[all_labels_np == 0] == 0).mean() if (all_labels_np == 0).sum() > 0 else 0
        dys_acc = (all_preds_np[all_labels_np == 1] == 1).mean() if (all_labels_np == 1).sum() > 0 else 0

        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"  >> [TRAIN] Control acc={ctrl_acc:.2%}, Dysarthric acc={dys_acc:.2%}")

        return avg_loss, accuracy, f1

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} [Val]")

            for audio, labels, metadata in pbar:
                audio = audio.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(audio)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Statistics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5

        # Per-class accuracy
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        ctrl_acc = (all_preds_np[all_labels_np == 0] == 0).mean() if (all_labels_np == 0).sum() > 0 else 0
        dys_acc = (all_preds_np[all_labels_np == 1] == 1).mean() if (all_labels_np == 1).sum() > 0 else 0

        pred_ctrl = (all_preds_np == 0).sum()
        pred_dys = (all_preds_np == 1).sum()

        print(f"Epoch {epoch+1} - Val Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"  >> [VAL] Control acc={ctrl_acc:.2%}, Dysarthric acc={dys_acc:.2%}")
        print(f"  >> [VAL] Predictions: {pred_ctrl} control, {pred_dys} dysarthric")

        if ctrl_acc < 0.2 or dys_acc < 0.2:
            print(f"  ⚠️ WARNING: Model collapsing! One class < 20% accuracy")

        # Tensorboard logging
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Val/F1', f1, epoch)
        self.writer.add_scalar('Val/AUC', auc, epoch)

        return avg_loss, accuracy, f1, auc, ctrl_acc, dys_acc

    def train(self):
        """Full training loop"""
        print(f"\nStarting training...")

        for epoch in range(self.args.num_epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_f1, val_auc, ctrl_acc, dys_acc = self.validate(epoch)

            # Calculate balanced accuracy
            balanced_acc = (ctrl_acc + dys_acc) / 2
            train_val_gap = train_acc - val_acc

            print(f"  >>> Balanced Acc: {balanced_acc:.4f} | Gap: {train_val_gap:.4f}")

            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'train/f1': train_f1,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'val/f1': val_f1,
                    'val/auc': val_auc,
                    'val/control_acc': ctrl_acc,
                    'val/dysarthric_acc': dys_acc,
                    'val/balanced_acc': balanced_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            # Scheduler step
            self.scheduler.step()

            # Save best model (by balanced accuracy)
            if balanced_acc > self.best_val_acc:
                self.best_val_acc = balanced_acc
                self.save_checkpoint(epoch, balanced_acc, is_best=True)
                print(f"  >>> New best model! Balanced Acc: {balanced_acc:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, balanced_acc, is_best=False)

        print(f"\nTraining completed! Best balanced accuracy: {self.best_val_acc:.4f}")
        self.writer.close()

        if self.args.use_wandb:
            wandb.finish()

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'args': vars(self.args)
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'

        torch.save(checkpoint, path)

    def test(self, checkpoint_path=None):
        """Test the model"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'

        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for audio, labels, metadata in tqdm(self.test_loader, desc="Testing"):
                audio = audio.to(self.device)
                logits = self.model(audio)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_preds)

        # Per-class
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        ctrl_acc = (all_preds_np[all_labels_np == 0] == 0).mean()
        dys_acc = (all_preds_np[all_labels_np == 1] == 1).mean()
        balanced_acc = (ctrl_acc + dys_acc) / 2

        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"\nPer-class accuracy:")
        print(f"  Control: {ctrl_acc:.2%}")
        print(f"  Dysarthric: {dys_acc:.2%}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print("=" * 50)

        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(checkpoint_path),
            'metrics': {
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc),
                'control_acc': float(ctrl_acc),
                'dysarthric_acc': float(dys_acc)
            },
            'confusion_matrix': cm.tolist(),
            'config': vars(self.args)
        }

        # Save results to JSON
        results_dir = self.output_dir / 'results'
        results_dir.mkdir(exist_ok=True)

        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f'test_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

        # Also save as latest results
        latest_file = results_dir / 'test_results_latest.json'
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Latest results saved to: {latest_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Train Wav2Vec2 Dysarthria Classifier')

    # Data
    parser.add_argument('--data_root', type=str, default='./TORGO_database')
    parser.add_argument('--output_dir', type=str, default='./output/wav2vec2')
    parser.add_argument('--max_length_sec', type=float, default=4.0, help='Max audio length in seconds')
    parser.add_argument('--split_mode', type=str, default='speaker', choices=['speaker', 'sample'],
                        help='speaker: fair evaluation (separate speakers), sample: quick dev only (mixed speakers)')

    # Model
    parser.add_argument('--model_type', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert'])
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base')
    parser.add_argument('--freeze_encoder', action='store_true', default=False, help='Freeze encoder (not recommended for dysarthric speech)')
    parser.add_argument('--unfreeze_encoder', action='store_true', help='Unfreeze encoder for fine-tuning (default now)')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'first', 'attention'])
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--batch_size', type=int, default=8, help='Smaller batch for Wav2Vec2')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_interval', type=int, default=5)

    # Wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='dysarthria-wav2vec2')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'])
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()

    # Handle freeze_encoder flag
    if args.unfreeze_encoder:
        args.freeze_encoder = False

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Train
    trainer = Wav2Vec2Trainer(args)

    if args.mode in ['train', 'both']:
        trainer.train()

    if args.mode in ['test', 'both']:
        trainer.test(checkpoint_path=args.checkpoint)


if __name__ == '__main__':
    main()
