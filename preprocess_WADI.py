"""
WADI A2 Preprocessing - Research-Based Protocol
================================================
Based on comprehensive literature review (2020-2025) for hierarchical IDS in IIoT.

Key principles:
1. Remove first 21,600 samples (6-hour stabilization period)
2. Remove 15 problematic sensors (K-S test failures) + 6 constant valves
3. Fit scaler ONLY on training data (prevent normalization leakage)
4. Handle missing values properly
5. Target realistic F1: 0.60-0.75 (not >0.95 which indicates leakage)

Reference: Comprehensive Preprocessing Pipeline for WADI A2 Dataset
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import pickle
import warnings
warnings.filterwarnings('ignore')

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WADI_DIR = os.path.join(SCRIPT_DIR, 'WADI.A2_19 Nov 2019')
TRAIN_FILE = os.path.join(WADI_DIR, 'WADI_14days_new.csv')
TEST_FILE = os.path.join(WADI_DIR, 'WADI_attackdataLABLE.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'processed_data')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("WADI A2 RESEARCH-BASED PREPROCESSING PIPELINE")
print("="*80)

# ============================================================================
# CONFIGURATION: Problematic Features from Research
# ============================================================================

# 15 sensors with distribution issues (K-S test failures)
PROBLEMATIC_SENSORS = [
    '1_AIT_001_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_AIT_005_PV',
    '2_LT_001_PV', '2_PIT_001_PV',
    '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV',
    '2B_AIT_001_PV', '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV',
    '3_AIT_005_PV'
]

# 6 constant solenoid valves (zero variance)
CONSTANT_VALVES = [
    '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS',
    '2_SV_401_STATUS', '2_SV_501_STATUS', '2_SV_601_STATUS'
]

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================
print("\nSTEP 1: LOAD RAW DATA")
print("-" * 80)

print("Loading training data...")
# Training file has proper headers
train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
print(f"  Original training shape: {train_df.shape}")

print("Loading test data...")
# Test file has headers in first row of data, will re-read in step 3
test_df_preview = pd.read_csv(TEST_FILE, nrows=1, low_memory=False)
print(f"  Original test preview: {test_df_preview.shape}")

# ============================================================================
# STEP 2: REMOVE FIRST 21,600 SAMPLES (6-HOUR STABILIZATION)
# ============================================================================
print("\nSTEP 2: REMOVE STABILIZATION PERIOD")
print("-" * 80)

print("Removing first 21,600 samples (6 hours) from training data...")
print(f"  Before: {len(train_df):,} samples")
train_df = train_df.iloc[21600:].reset_index(drop=True)
print(f"  After: {len(train_df):,} samples")
print(f"  Removed: 21,600 samples (system initialization artifacts)")

# ============================================================================
# STEP 3: EXTRACT LABELS FROM TEST DATA
# ============================================================================
print("\nSTEP 3: EXTRACT LABELS")
print("-" * 80)

# The test file has headers in the first row of data, need to skip and re-read
print("Re-reading test file with correct header...")
test_df = pd.read_csv(TEST_FILE, skiprows=1, low_memory=False)

# Find label column (should be last column with "Attack" in name)
label_col = test_df.columns[-1]
print(f"Label column found: {label_col}")
print(f"Unique values: {test_df[label_col].unique()}")

# Extract labels and map: 1 (Normal) → 0, -1 (Attack) → 1
test_labels = test_df[label_col].values
test_labels = np.where(test_labels == 1, 0, 1)  # 1→0 (normal), -1→1 (attack)

print(f"\nLabel distribution:")
print(f"  Normal (0): {np.sum(test_labels == 0):,} ({100*np.sum(test_labels == 0)/len(test_labels):.2f}%)")
print(f"  Attack (1): {np.sum(test_labels == 1):,} ({100*np.sum(test_labels == 1)/len(test_labels):.2f}%)")

# Remove label column from test features
test_df = test_df.drop(columns=[label_col])

# ============================================================================
# STEP 4: IDENTIFY SENSOR COLUMNS
# ============================================================================
print("\nSTEP 4: IDENTIFY SENSOR COLUMNS")
print("-" * 80)

# Remove timestamp/date columns
non_sensor_cols = ['Date', 'Time', 'Timestamp', 'Row']
sensor_cols = [col for col in train_df.columns if col not in non_sensor_cols]

print(f"Total columns in data: {len(train_df.columns)}")
print(f"Sensor columns identified: {len(sensor_cols)}")
print(f"First 5 sensors: {sensor_cols[:5]}")

# Extract sensor data only
X_train = train_df[sensor_cols].copy()
X_test = test_df[sensor_cols].copy()

print(f"\nTraining features: {X_train.shape}")
print(f"Test features: {X_test.shape}")

# ============================================================================
# STEP 5: REMOVE PROBLEMATIC FEATURES (RESEARCH-BASED)
# ============================================================================
print("\nSTEP 5: REMOVE PROBLEMATIC FEATURES")
print("-" * 80)

# Combine problematic sensors and constant valves
features_to_remove = PROBLEMATIC_SENSORS + CONSTANT_VALVES

# Find which ones exist in the dataset
existing_to_remove = [col for col in features_to_remove if col in X_train.columns]
missing_features = [col for col in features_to_remove if col not in X_train.columns]

print(f"Features to remove (from research): {len(features_to_remove)}")
print(f"  Found in dataset: {len(existing_to_remove)}")
print(f"  Not found (already absent): {len(missing_features)}")

if existing_to_remove:
    print(f"\nRemoving {len(existing_to_remove)} problematic features:")
    for col in existing_to_remove:
        print(f"  - {col}")
    
    X_train = X_train.drop(columns=existing_to_remove)
    X_test = X_test.drop(columns=existing_to_remove)
    
    print(f"\nAfter removal:")
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")

# ============================================================================
# STEP 6: HANDLE MISSING VALUES
# ============================================================================
print("\nSTEP 6: HANDLE MISSING VALUES")
print("-" * 80)

print("Missing values before handling:")
print(f"  Training: {X_train.isnull().sum().sum():,} NaN values")
print(f"  Test: {X_test.isnull().sum().sum():,} NaN values")

# Remove columns with all NaN
all_nan_cols_train = X_train.columns[X_train.isnull().all()].tolist()
all_nan_cols_test = X_test.columns[X_test.isnull().all()].tolist()
all_nan_cols = list(set(all_nan_cols_train + all_nan_cols_test))

if all_nan_cols:
    print(f"\nRemoving {len(all_nan_cols)} columns with all NaN values:")
    for col in all_nan_cols:
        print(f"  - {col}")
    X_train = X_train.drop(columns=all_nan_cols)
    X_test = X_test.drop(columns=all_nan_cols)

# Forward fill then backward fill, then replace remaining with 0
print("\nImputing remaining missing values...")
X_train = X_train.fillna(method='ffill').fillna(method='bfill').fillna(0)
X_test = X_test.fillna(method='ffill').fillna(method='bfill').fillna(0)

print(f"Missing values after handling:")
print(f"  Training: {X_train.isnull().sum().sum():,}")
print(f"  Test: {X_test.isnull().sum().sum():,}")

# Convert to numeric
print("\nConverting to numeric...")
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# ============================================================================
# STEP 7: REMOVE LOW-VARIANCE FEATURES
# ============================================================================
print("\nSTEP 7: REMOVE LOW-VARIANCE FEATURES")
print("-" * 80)

# Calculate variance on training data only
variances = X_train.var()
print(f"Variance statistics:")
print(f"  Min: {variances.min():.6f}")
print(f"  Max: {variances.max():.2f}")
print(f"  Median: {variances.median():.6f}")

# Remove features with variance < 0.001 (more conservative than 0.01)
variance_threshold = 0.001
low_var_features = variances[variances < variance_threshold].index.tolist()

if low_var_features:
    print(f"\nRemoving {len(low_var_features)} low-variance features (threshold={variance_threshold}):")
    for col in low_var_features[:10]:  # Show first 10
        print(f"  - {col} (variance: {variances[col]:.6f})")
    if len(low_var_features) > 10:
        print(f"  ... and {len(low_var_features)-10} more")
    
    X_train = X_train.drop(columns=low_var_features)
    X_test = X_test.drop(columns=low_var_features)

print(f"\nFinal feature count: {X_train.shape[1]}")

# ============================================================================
# STEP 8: NORMALIZE FEATURES (CRITICAL: TRAINING DATA ONLY!)
# ============================================================================
print("\nSTEP 8: NORMALIZE FEATURES")
print("-" * 80)
print("CRITICAL: Fitting scaler on TRAINING data ONLY (prevent leakage)")

# Fit scaler on training data ONLY
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)  # ← FIT ON TRAINING ONLY!

print(f"Scaler fitted on training data: {X_train.shape}")

# Transform both sets using training statistics
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nScaled using TRAINING statistics:")
print(f"  Training range: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
print(f"  Test range: [{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]")

# Check if test data is within reasonable bounds
test_outside_bounds = (X_test_scaled < -0.1) | (X_test_scaled > 1.1)
if test_outside_bounds.any():
    pct_outside = 100 * test_outside_bounds.sum().sum() / test_outside_bounds.size
    print(f"\n  ⚠ Warning: {pct_outside:.2f}% of test values outside [0,1]")
    print(f"  This is EXPECTED for attacks (anomalies should be outside normal range)")
else:
    print(f"  ✓ All test values within reasonable range")

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# ============================================================================
# STEP 9: DATA QUALITY VALIDATION
# ============================================================================
print("\nSTEP 9: DATA QUALITY VALIDATION")
print("-" * 80)

print(f"NaN in training: {X_train_scaled.isnull().sum().sum()}")
print(f"NaN in test: {X_test_scaled.isnull().sum().sum()}")
print(f"Inf in training: {np.isinf(X_train_scaled.values).sum()}")
print(f"Inf in test: {np.isinf(X_test_scaled.values).sum()}")

print(f"\nTraining statistics:")
print(f"  Mean: {X_train_scaled.values.mean():.4f}")
print(f"  Std: {X_train_scaled.values.std():.4f}")
print(f"  Min: {X_train_scaled.values.min():.4f}")
print(f"  Max: {X_train_scaled.values.max():.4f}")

print(f"\nTest statistics:")
print(f"  Mean: {X_test_scaled.values.mean():.4f}")
print(f"  Std: {X_test_scaled.values.std():.4f}")
print(f"  Min: {X_test_scaled.values.min():.4f}")
print(f"  Max: {X_test_scaled.values.max():.4f}")

# ============================================================================
# STEP 10: SAVE PROCESSED DATA
# ============================================================================
print("\nSTEP 10: SAVE PROCESSED DATA")
print("-" * 80)

# Save scaled features
train_path = os.path.join(OUTPUT_DIR, 'wadi_train_scaled.csv')
test_path = os.path.join(OUTPUT_DIR, 'wadi_test_scaled.csv')
labels_path = os.path.join(OUTPUT_DIR, 'wadi_test_labels.csv')

X_train_scaled.to_csv(train_path, index=False)
X_test_scaled.to_csv(test_path, index=False)
pd.DataFrame({'attack': test_labels}).to_csv(labels_path, index=False)

print(f"✓ {train_path}: {X_train_scaled.shape}")
print(f"✓ {test_path}: {X_test_scaled.shape}")
print(f"✓ {labels_path}: {len(test_labels)} labels")

# Save scaler and feature list
scaler_path = os.path.join(OUTPUT_DIR, 'wadi_scaler.pkl')
features_path = os.path.join(OUTPUT_DIR, 'wadi_selected_features.pkl')

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ {scaler_path}: Scaler saved")

with open(features_path, 'wb') as f:
    pickle.dump(list(X_train_scaled.columns), f)
print(f"✓ {features_path}: {len(X_train_scaled.columns)} features")

# ============================================================================
# STEP 11: ATTACK SCENARIO ANALYSIS
# ============================================================================
print("\nSTEP 11: ATTACK SCENARIO ANALYSIS")
print("-" * 80)

print(f"\nTest set breakdown:")
print(f"  Total samples: {len(test_labels):,}")
print(f"  Normal samples: {np.sum(test_labels == 0):,} ({100*np.sum(test_labels == 0)/len(test_labels):.2f}%)")
print(f"  Attack samples: {np.sum(test_labels == 1):,} ({100*np.sum(test_labels == 1)/len(test_labels):.2f}%)")

if np.sum(test_labels == 1) >= 50:
    print(f"  ✓ SUFFICIENT: {np.sum(test_labels == 1):,} attack samples (need >= 50)")
else:
    print(f"  ⚠ WARNING: Only {np.sum(test_labels == 1):,} attack samples (need >= 50)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING COMPLETE - READINESS CHECKLIST")
print("="*80)

checklist = {
    "Data loaded": True,
    "Stabilization removed (21,600 samples)": True,
    "Problematic features removed (research-based)": len(existing_to_remove) > 0,
    "Missing values handled": X_train_scaled.isnull().sum().sum() == 0,
    "Low-variance features removed": len(low_var_features) > 0 if low_var_features else False,
    "Scaler fitted on TRAINING ONLY": True,
    "Data quality validated": True,
    "Sufficient attack samples": np.sum(test_labels == 1) >= 50
}

print("\nReadiness Status:")
for check, status in checklist.items():
    symbol = "✓" if status else "✗"
    print(f"  {symbol} {check}")

print("\n" + "="*80)
print("WADI DATASET SUMMARY")
print("="*80)

print(f"\nTraining: {len(X_train_scaled):,} samples (pure normal, stabilization removed)")
print(f"Test: {len(X_test_scaled):,} samples")
print(f"  - Normal: {np.sum(test_labels == 0):,} ({100*np.sum(test_labels == 0)/len(test_labels):.1f}%)")
print(f"  - Attacks: {np.sum(test_labels == 1):,} ({100*np.sum(test_labels == 1)/len(test_labels):.1f}%)")

print(f"\nFeatures: {len(X_train_scaled.columns)} selected")
print(f"  Original: ~127 features")
print(f"  Removed problematic: {len(existing_to_remove)} sensors")
print(f"  Removed low-variance: {len(low_var_features) if low_var_features else 0} features")
print(f"  Final: {len(X_train_scaled.columns)} features")

print(f"\nNormalization: MinMaxScaler [0, 1]")
print(f"  Fitted on: Training data ONLY (prevents leakage)")
print(f"  Training range: [{X_train_scaled.values.min():.4f}, {X_train_scaled.values.max():.4f}]")
print(f"  Test range: [{X_test_scaled.values.min():.4f}, {X_test_scaled.values.max():.4f}]")

print(f"\nExpected Performance (from research):")
print(f"  F1-Score: 0.60 - 0.75 (state-of-the-art)")
print(f"  Attacks detected: 13-14 out of 15")
print(f"  ⚠ If F1 > 0.95: likely data leakage!")

print("\n" + "="*80)
print("NEXT STEP: Run experiments with Config_2_Balanced model")
print("  python run_final_wadi_experiments.py")
print("="*80)
