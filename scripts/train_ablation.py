"""
Training Script for Ablation Studies
Supports extended model configurations for pooling, layer-wise analysis, etc.
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
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import ablation model
ablation_path = Path(__file__).parent.parent / "models" / "Wav2Vec2ClassifierAblation.py"
spec = importlib.util.spec_from_file_location("Wav2Vec2ClassifierAblation", ablation_path)
ablation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation_module)
Wav2Vec2AblationClassifier = ablation_module.Wav2Vec2AblationClassifier
LayerWiseClassifier = ablation_module.LayerWiseClassifier

from datasets_process.torgo_raw import create_torgo_raw_dataloaders


class AblationTrainer:
    """Trainer for ablation studies"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "=" * 60)
        print("ABLATION STUDY TRAINER")
        print("=" * 60)
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("=" * 60 + "\n")

        # Output directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))

        # Load data
        print("Loading TORGO dataset...")
        self.train_loader, self.val_loader, self.test_loader = create_torgo_raw_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            max_length_sec=args.max_length_sec,
            num_workers=args.num_workers,
            seed=args.seed,
            split_mode=args.split_mode
        )

        # Initialize model
        print("\nInitializing model...")
        if args.layer_wise:
            # Layer-wise probing (frozen encoder)
            self.model = LayerWiseClassifier(
                model_name=args.model_name,
                num_classes=2,
                dropout=args.dropout,
                layer_index=args.use_layer,
                model_type=args.model_type
            )
        else:
            # Standard ablation model
            self.model = Wav2Vec2AblationClassifier(
                model_name=args.model_name,
                num_classes=2,
                dropout=args.dropout,
                freeze_encoder=args.freeze_encoder,
                freeze_feature_extractor=True,
                pooling=args.pooling,
                use_layer=args.use_layer,
                model_type=args.model_type
            )
        self.model = self.model.to(self.device)

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs,
            eta_min=1e-6
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.best_val_acc = 0.0
        self.global_step = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} [Train]")

        for audio, labels, metadata in pbar:
            audio = audio.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(audio)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})
            self.global_step += 1

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
        return avg_loss, accuracy, f1

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for audio, labels, metadata in tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]"):
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

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5

        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        ctrl_acc = (all_preds_np[all_labels_np == 0] == 0).mean() if (all_labels_np == 0).sum() > 0 else 0
        dys_acc = (all_preds_np[all_labels_np == 1] == 1).mean() if (all_labels_np == 1).sum() > 0 else 0
        balanced_acc = (ctrl_acc + dys_acc) / 2

        print(f"Epoch {epoch+1} - Val Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Bal.Acc: {balanced_acc:.4f}")

        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Val/F1', f1, epoch)
        self.writer.add_scalar('Val/BalancedAcc', balanced_acc, epoch)

        return avg_loss, accuracy, f1, auc, balanced_acc

    def train(self):
        print(f"\nStarting training...")

        for epoch in range(self.args.num_epochs):
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            val_loss, val_acc, val_f1, val_auc, balanced_acc = self.validate(epoch)

            self.scheduler.step()

            if balanced_acc > self.best_val_acc:
                self.best_val_acc = balanced_acc
                self.save_checkpoint(epoch, balanced_acc, is_best=True)
                print(f"  >>> New best model! Balanced Acc: {balanced_acc:.4f}")

        print(f"\nTraining completed! Best balanced accuracy: {self.best_val_acc:.4f}")
        self.writer.close()

    def save_checkpoint(self, epoch, val_acc, is_best=False):
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
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'

        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for audio, labels, metadata in tqdm(self.test_loader, desc="Testing"):
                audio = audio.to(self.device)
                logits = self.model(audio)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_preds)

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
        print(f"\nConfusion Matrix:\n{cm}")
        print("=" * 50)

        # Save results
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

        results_dir = self.output_dir / 'results'
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f'test_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        latest_file = results_dir / 'test_results_latest.json'
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Training')

    # Data
    parser.add_argument('--data_root', type=str, default='./TORGO_database')
    parser.add_argument('--output_dir', type=str, default='./output/ablation')
    parser.add_argument('--max_length_sec', type=float, default=4.0)
    parser.add_argument('--split_mode', type=str, default='speaker', choices=['speaker', 'sample'])

    # Model
    parser.add_argument('--model_type', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert'])
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base')
    parser.add_argument('--freeze_encoder', action='store_true', default=False)
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'first', 'attention', 'max', 'weighted_sum'])
    parser.add_argument('--use_layer', type=int, default=-1,
                        help='Which layer to use (-1 for last, 0-11 for specific)')
    parser.add_argument('--layer_wise', action='store_true',
                        help='Enable layer-wise probing (frozen encoder)')
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'])
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer = AblationTrainer(args)

    if args.mode in ['train', 'both']:
        trainer.train()

    if args.mode in ['test', 'both']:
        trainer.test(checkpoint_path=args.checkpoint)


if __name__ == '__main__':
    main()
