"""
WADI Dataset Utilities for SCADA Anomaly Detection

WADI (Water Distribution) Dataset:
- 127 sensor features
- 962,745 training samples (normal operation)
- 172,801 test samples (with attacks)
- Sampling rate: 1 second

Data Loading Pipeline:
1. Load scaled CSV files (already normalized)
2. Create sliding windows (4 timesteps)
3. Generate PyTorch DataLoaders
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings


class WADIDataset(Dataset):
    """
    PyTorch Dataset for WADI time series data
    
    Creates sliding windows from sensor readings for sequence models
    """
    
    def __init__(self, data, labels, seq_len=4, stride=1):
        """
        Args:
            data: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,) with 0=normal, 1=attack
            seq_len: Number of timesteps in each window (default: 4)
            stride: Step size between windows (default: 1)
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        self.seq_len = seq_len
        self.stride = stride
        
        # Calculate number of valid windows
        self.n_samples = (len(data) - seq_len) // stride + 1
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        
        # Get sequence window
        x = self.data[start:end]  # (seq_len, n_features)
        
        # Label is based on last timestep in window
        y = self.labels[end - 1].unsqueeze(0)  # (1,)
        
        return x, y


def load_wadi_data(data_dir='processed_data', include_raw=False):
    """
    Load preprocessed WADI dataset
    
    Args:
        data_dir: Directory containing processed CSV files
        include_raw: If True, also return raw (unscaled) data
    
    Returns:
        dict with train_data, train_labels, test_data, test_labels
    """
    print("Loading WADI dataset...")
    
    # Load scaled data
    train_data = pd.read_csv(os.path.join(data_dir, 'wadi_train_scaled.csv')).values
    test_data = pd.read_csv(os.path.join(data_dir, 'wadi_test_scaled.csv')).values
    
    # Load labels
    train_labels = pd.read_csv(os.path.join(data_dir, 'wadi_train_labels.csv')).values.ravel()
    test_labels = pd.read_csv(os.path.join(data_dir, 'wadi_test_labels.csv')).values.ravel()
    
    print(f"  Train: {train_data.shape[0]:,} samples, {train_data.shape[1]} features")
    print(f"  Test: {test_data.shape[0]:,} samples, {test_data.shape[1]} features")
    print(f"  Attack ratio (test): {test_labels.mean()*100:.2f}%")
    
    result = {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'n_features': train_data.shape[1]
    }
    
    if include_raw:
        result['train_raw'] = pd.read_csv(os.path.join(data_dir, 'wadi_train_raw.csv')).values
        result['test_raw'] = pd.read_csv(os.path.join(data_dir, 'wadi_test_raw.csv')).values
    
    return result


def create_dataloaders(train_data, train_labels, test_data, test_labels,
                       seq_len=4, batch_size=64, val_split=0.1, num_workers=0):
    """
    Create PyTorch DataLoaders for training, validation, and testing
    
    Args:
        train_data: Training features
        train_labels: Training labels
        test_data: Test features
        test_labels: Test labels
        seq_len: Sequence length for sliding windows
        batch_size: Batch size for DataLoaders
        val_split: Fraction of training data for validation
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split training into train/val (temporal split - no data leakage)
    n_train = len(train_data)
    n_val = int(n_train * val_split)
    n_train_actual = n_train - n_val
    
    # Use last portion as validation (temporal integrity)
    train_split_data = train_data[:n_train_actual]
    train_split_labels = train_labels[:n_train_actual]
    val_split_data = train_data[n_train_actual:]
    val_split_labels = train_labels[n_train_actual:]
    
    print(f"  Train split: {len(train_split_data):,} samples")
    print(f"  Val split: {len(val_split_data):,} samples")
    print(f"  Test: {len(test_data):,} samples")
    
    # Create datasets
    train_dataset = WADIDataset(train_split_data, train_split_labels, seq_len=seq_len)
    val_dataset = WADIDataset(val_split_data, val_split_labels, seq_len=seq_len)
    test_dataset = WADIDataset(test_data, test_labels, seq_len=seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def preprocess_raw_data(raw_data_path, output_dir='processed_data'):
    """
    Preprocess raw WADI CSV files
    
    Steps:
    1. Load raw CSV
    2. Handle missing values
    3. Normalize using StandardScaler (fit on train only)
    4. Save processed files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Preprocessing raw WADI data...")
    
    # Load raw files
    train_df = pd.read_csv(os.path.join(raw_data_path, 'WADI_14days_new.csv'))
    test_df = pd.read_csv(os.path.join(raw_data_path, 'WADI_attackdataLABLE.csv'))
    
    # Extract features (exclude timestamp and label columns)
    feature_cols = [c for c in train_df.columns if c not in ['Row', 'Date', 'Time', 'Attack LABLE (1:No Attack, -1:Attack)']]
    
    train_features = train_df[feature_cols].values
    test_features = test_df[feature_cols].values
    
    # Handle missing values
    train_features = np.nan_to_num(train_features, nan=0.0)
    test_features = np.nan_to_num(test_features, nan=0.0)
    
    # Extract labels (1=normal, -1=attack in raw data)
    train_labels = np.zeros(len(train_features))  # All normal in training
    test_labels = (test_df['Attack LABLE (1:No Attack, -1:Attack)'].values == -1).astype(float)
    
    # Fit scaler on training data only (no data leakage)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    
    # Save processed data
    pd.DataFrame(train_scaled).to_csv(os.path.join(output_dir, 'wadi_train_scaled.csv'), index=False)
    pd.DataFrame(test_scaled).to_csv(os.path.join(output_dir, 'wadi_test_scaled.csv'), index=False)
    pd.DataFrame(train_labels).to_csv(os.path.join(output_dir, 'wadi_train_labels.csv'), index=False)
    pd.DataFrame(test_labels).to_csv(os.path.join(output_dir, 'wadi_test_labels.csv'), index=False)
    
    # Also save raw (unscaled) for reference
    pd.DataFrame(train_features).to_csv(os.path.join(output_dir, 'wadi_train_raw.csv'), index=False)
    pd.DataFrame(test_features).to_csv(os.path.join(output_dir, 'wadi_test_raw.csv'), index=False)
    
    print(f"  Saved {len(train_features):,} train samples")
    print(f"  Saved {len(test_features):,} test samples")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Attack ratio (test): {test_labels.mean()*100:.2f}%")
    
    return scaler


def get_sample_batch(data_dir='processed_data', batch_size=32, seq_len=4):
    """
    Get a sample batch for testing/debugging
    """
    data = load_wadi_data(data_dir)
    train_loader, _, _ = create_dataloaders(
        data['train_data'], data['train_labels'],
        data['test_data'], data['test_labels'],
        seq_len=seq_len, batch_size=batch_size
    )
    
    for x, y in train_loader:
        return x, y


if __name__ == "__main__":
    # Test data loading
    print("Testing WADI data loading...")
    
    try:
        data = load_wadi_data('processed_data')
        
        train_loader, val_loader, test_loader = create_dataloaders(
            data['train_data'], data['train_labels'],
            data['test_data'], data['test_labels'],
            seq_len=4, batch_size=64
        )
        
        # Get sample batch
        for x, y in train_loader:
            print(f"\n✓ Sample batch:")
            print(f"  X shape: {x.shape}")  # (batch, seq_len, features)
            print(f"  Y shape: {y.shape}")  # (batch, 1)
            print(f"  Y values: {y[:5].flatten().numpy()}")
            break
            
        print(f"\n✓ DataLoaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
    except FileNotFoundError as e:
        print(f"⚠ Data files not found. Run preprocess_WADI.py first.")
        print(f"  Error: {e}")
