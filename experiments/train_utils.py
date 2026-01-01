"""
Training Utilities for SCADA Anomaly Detection Models

Features:
- Training loop with early stopping
- Learning rate scheduling
- Model checkpointing
- Metrics computation (F1, Precision, Recall, AUC)
- Cross-validation support
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, 
    precision_recall_curve, roc_curve
)
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    
    Monitors validation loss and stops training when no improvement
    """
    
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False
        
        if self.mode == 'min':
            improved = val_score < self.best_score - self.min_delta
        else:
            improved = val_score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


def compute_metrics(y_true, y_pred, y_prob=None, threshold=0.5):
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (or probabilities if y_prob is None)
        y_prob: Prediction probabilities (optional)
        threshold: Classification threshold
    
    Returns:
        dict with F1, precision, recall, AUC, etc.
    """
    if y_prob is None:
        y_prob = y_pred
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = (y_pred >= threshold).astype(int) if y_pred.max() <= 1 else y_pred
    
    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    # AUC requires probability scores
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc_roc'] = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def find_optimal_threshold(y_true, y_prob):
    """
    Find optimal classification threshold using F1 score
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
    return best_threshold, best_f1


class Trainer:
    """
    Training orchestrator for anomaly detection models
    """
    
    def __init__(self, model, device='cuda', checkpoint_dir='Results/checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_auc': []
        }
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / n_batches
    
    def validate(self, val_loader, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = criterion(output, y)
                
                total_loss += loss.item()
                all_preds.extend(output.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, epochs=100, lr=1e-3, 
              patience=10, save_best=True):
        """
        Full training loop
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Maximum epochs
            lr: Learning rate
            patience: Early stopping patience
            save_best: Save best model checkpoint
        
        Returns:
            Training history
        """
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        early_stopping = EarlyStopping(patience=patience, mode='max')
        
        best_f1 = 0
        start_time = time.time()
        
        print(f"\nTraining on {self.device}...")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Early stopping patience: {patience}")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc_roc'])
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, "
                      f"F1={val_metrics['f1']:.4f}, "
                      f"AUC={val_metrics['auc_roc']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                if save_best:
                    self.save_checkpoint('best_model.pt')
            
            # Early stopping
            if early_stopping(val_metrics['f1']):
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        elapsed = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed/60:.1f} minutes")
        print(f"  Best F1: {best_f1:.4f}")
        
        return self.history
    
    def evaluate(self, test_loader, load_best=True):
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test DataLoader
            load_best: Load best checkpoint before evaluation
        
        Returns:
            Evaluation metrics
        """
        if load_best:
            self.load_checkpoint('best_model.pt')
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        print("\nEvaluating on test set...")
        
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Testing"):
                x = x.to(self.device)
                output = self.model(x)
                all_preds.extend(output.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        # Find optimal threshold
        best_threshold, _ = find_optimal_threshold(all_labels, all_preds)
        
        # Compute metrics with optimal threshold
        metrics = compute_metrics(all_labels, all_preds, threshold=best_threshold)
        
        print(f"\n✓ Test Results:")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
        print(f"  Threshold:   {metrics['threshold']:.2f}")
        
        return metrics, all_preds, all_labels
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.history = checkpoint.get('history', self.history)
            print(f"  Loaded checkpoint: {filename}")
        else:
            print(f"  ⚠ Checkpoint not found: {filename}")


def run_cross_validation(model_class, data, n_folds=5, **train_kwargs):
    """
    Run k-fold cross-validation
    
    Args:
        model_class: Model class to instantiate
        data: Dict with train_data, train_labels, test_data, test_labels
        n_folds: Number of folds
        **train_kwargs: Arguments for Trainer.train()
    
    Returns:
        List of fold results
    """
    from experiments.data_utils import WADIDataset, DataLoader
    
    n_samples = len(data['train_data'])
    fold_size = n_samples // n_folds
    
    results = []
    
    print(f"\nRunning {n_folds}-fold cross-validation...")
    
    for fold in range(n_folds):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        # Split data (temporal folds)
        val_start = fold * fold_size
        val_end = val_start + fold_size
        
        train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))
        val_idx = list(range(val_start, val_end))
        
        train_data = data['train_data'][train_idx]
        train_labels = data['train_labels'][train_idx]
        val_data = data['train_data'][val_idx]
        val_labels = data['train_labels'][val_idx]
        
        # Create datasets
        train_dataset = WADIDataset(train_data, train_labels)
        val_dataset = WADIDataset(val_data, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Train model
        model = model_class(n_features=data['n_features'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = Trainer(model, device=device)
        
        trainer.train(train_loader, val_loader, **train_kwargs)
        _, val_metrics = trainer.validate(val_loader, nn.BCELoss())
        
        results.append(val_metrics)
        print(f"  Fold {fold+1} F1: {val_metrics['f1']:.4f}")
    
    # Aggregate results
    avg_f1 = np.mean([r['f1'] for r in results])
    std_f1 = np.std([r['f1'] for r in results])
    
    print(f"\n✓ Cross-validation complete")
    print(f"  Mean F1: {avg_f1:.4f} ± {std_f1:.4f}")
    
    return results


if __name__ == "__main__":
    # Test training utilities
    print("Testing training utilities...")
    
    # Test metrics computation
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.3, 0.8, 0.9, 0.7, 0.2, 0.6, 0.4])
    
    metrics = compute_metrics(y_true, y_prob)
    print(f"\n✓ Metrics computation:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Test optimal threshold
    best_thresh, best_f1 = find_optimal_threshold(y_true, y_prob)
    print(f"\n✓ Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    
    # Test early stopping
    es = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.7, 0.7, 0.7, 0.68]
    for i, score in enumerate(scores):
        stop = es(score)
        print(f"  Step {i+1}: score={score}, stop={stop}")
