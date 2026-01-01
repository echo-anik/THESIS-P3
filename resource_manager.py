"""
Adaptive Resource Manager for Hierarchical IDS Experiments
===========================================================
Automatically detects system resources and configures experiments accordingly.
Designed to work on both low-end (16GB RAM, RTX 3060 Ti) and high-end systems.

Features:
- Auto-detection of available RAM and CPU cores
- Dynamic batch size calculation
- Memory-efficient processing
- System profiling and recommendations
"""

import psutil
import platform
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class SystemProfile:
    """System hardware profile"""
    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    cpu_freq_mhz: float
    system_type: str  # 'low_end', 'mid_range', 'high_end'
    has_gpu: bool
    gpu_name: str

@dataclass
class ExperimentConfig:
    """Experiment configuration based on system resources"""
    batch_size: int
    chunk_size: int
    n_jobs: int
    edge_training_percentages: list
    max_samples_in_memory: int
    enable_parallel: bool
    checkpoint_frequency: int  # Save checkpoint every N experiments


class AdaptiveResourceManager:
    """Manages resource allocation based on available hardware"""

    def __init__(self):
        self.system_profile = self.detect_system()
        self.config = self.generate_config()

    def detect_system(self) -> SystemProfile:
        """Detect system hardware specifications"""

        # Memory info
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)

        # CPU info
        cpu_count = psutil.cpu_count(logical=True)
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_mhz = cpu_freq.current if cpu_freq else 0
        except:
            cpu_freq_mhz = 0

        # Classify system type
        if total_ram_gb < 20:
            system_type = 'low_end'
        elif total_ram_gb < 40:
            system_type = 'mid_range'
        else:
            system_type = 'high_end'

        # GPU detection (basic)
        has_gpu = False
        gpu_name = 'None'
        try:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
                gpu_name = torch.cuda.get_device_name(0)
        except:
            # Try alternative detection for NVIDIA
            if platform.system() == 'Windows':
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                           capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        has_gpu = True
                        gpu_name = result.stdout.strip()
                except:
                    pass

        return SystemProfile(
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            cpu_count=cpu_count,
            cpu_freq_mhz=cpu_freq_mhz,
            system_type=system_type,
            has_gpu=has_gpu,
            gpu_name=gpu_name
        )

    def generate_config(self) -> ExperimentConfig:
        """Generate experiment configuration based on system profile"""

        profile = self.system_profile

        # Configuration for different system types
        if profile.system_type == 'low_end':
            # Conservative settings for 16GB RAM systems
            config = ExperimentConfig(
                batch_size=1000,
                chunk_size=50000,
                n_jobs=max(1, profile.cpu_count // 2),  # Use half CPU cores
                edge_training_percentages=[0.05, 0.10, 0.15, 0.20],
                max_samples_in_memory=100000,
                enable_parallel=False,  # Process sequentially
                checkpoint_frequency=1  # Save after each experiment
            )

        elif profile.system_type == 'mid_range':
            # Balanced settings for 20-40GB RAM systems
            config = ExperimentConfig(
                batch_size=5000,
                chunk_size=100000,
                n_jobs=max(1, profile.cpu_count - 2),  # Leave 2 cores free
                edge_training_percentages=[0.05, 0.10, 0.15, 0.20, 0.25],
                max_samples_in_memory=500000,
                enable_parallel=True,
                checkpoint_frequency=2
            )

        else:  # high_end
            # Aggressive settings for 40+ GB RAM systems
            config = ExperimentConfig(
                batch_size=10000,
                chunk_size=200000,
                n_jobs=-1,  # Use all cores
                edge_training_percentages=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50],
                max_samples_in_memory=1000000,
                enable_parallel=True,
                checkpoint_frequency=5
            )

        return config

    def print_system_info(self):
        """Print detailed system information"""
        profile = self.system_profile
        config = self.config

        print("="*80)
        print(" SYSTEM HARDWARE PROFILE")
        print("="*80)
        print(f"System Type: {profile.system_type.upper()}")
        print(f"Total RAM: {profile.total_ram_gb:.1f} GB")
        print(f"Available RAM: {profile.available_ram_gb:.1f} GB")
        print(f"CPU Cores: {profile.cpu_count}")
        print(f"CPU Frequency: {profile.cpu_freq_mhz:.0f} MHz" if profile.cpu_freq_mhz > 0 else "CPU Frequency: N/A")
        print(f"GPU: {profile.gpu_name if profile.has_gpu else 'No GPU detected'}")
        print(f"Platform: {platform.system()} {platform.release()}")

        print("\n" + "="*80)
        print(" RECOMMENDED EXPERIMENT CONFIGURATION")
        print("="*80)
        print(f"Batch Size: {config.batch_size:,}")
        print(f"Chunk Size: {config.chunk_size:,}")
        print(f"Parallel Jobs: {config.n_jobs if config.n_jobs > 0 else 'All CPUs'}")
        print(f"Edge Training %: {[int(p*100) for p in config.edge_training_percentages]}")
        print(f"Max Samples in Memory: {config.max_samples_in_memory:,}")
        print(f"Parallel Processing: {'Enabled' if config.enable_parallel else 'Disabled (Sequential)'}")
        print(f"Checkpoint Frequency: Every {config.checkpoint_frequency} experiment(s)")

        print("\n" + "="*80)
        print(" OPTIMIZATION TIPS")
        print("="*80)

        if profile.system_type == 'low_end':
            print("Low-end system detected. Optimizations:")
            print("  - Checkpoint frequency set to maximum for data safety")
            print("  - Sequential processing to avoid memory issues")
            print("  - Reduced edge training percentages")
            print("  - Consider closing other applications during experiments")
            print("  - Estimated time: 2-4 hours per dataset")

        elif profile.system_type == 'mid_range':
            print("Mid-range system detected. Balanced configuration:")
            print("  - Parallel processing enabled")
            print("  - Good balance of speed and safety")
            print("  - Estimated time: 1-2 hours per dataset")

        else:
            print("High-end system detected. Aggressive optimization:")
            print("  - Full parallel processing")
            print("  - Extended edge training percentages for comprehensive analysis")
            print("  - Estimated time: 30-60 minutes per dataset")

        if not profile.has_gpu:
            print("\nNote: No GPU detected. All ML models will use CPU.")
            print("      This is fine for tree-based models (RandomForest, ExtraTrees)")

        print("="*80)

    def check_memory_for_dataset(self, n_samples: int, n_features: int) -> Tuple[bool, str]:
        """
        Check if dataset can fit in available memory

        Args:
            n_samples: Number of samples
            n_features: Number of features

        Returns:
            (can_fit, message)
        """
        # Estimate memory needed (rough calculation)
        # Each float64 value = 8 bytes
        estimated_mb = (n_samples * n_features * 8) / (1024**2)
        available_mb = self.system_profile.available_ram_gb * 1024

        # Leave 20% buffer
        safe_available = available_mb * 0.8

        can_fit = estimated_mb < safe_available

        if can_fit:
            msg = f"OK - Dataset fits in memory: {estimated_mb:.0f} MB needed, {safe_available:.0f} MB available"
        else:
            msg = f"WARNING - Dataset may cause memory issues: {estimated_mb:.0f} MB needed, {safe_available:.0f} MB available"
            msg += f"\n  Recommendation: Use chunked processing or reduce sample size"

        return can_fit, msg

    def get_optimal_sample_size(self, total_samples: int, target_percentage: float) -> int:
        """
        Calculate optimal sample size considering memory constraints

        Args:
            total_samples: Total available samples
            target_percentage: Desired percentage (e.g., 0.20 for 20%)

        Returns:
            Actual number of samples to use
        """
        target_samples = int(total_samples * target_percentage)

        # Cap at max_samples_in_memory if needed
        if target_samples > self.config.max_samples_in_memory:
            print(f"  ⚠ Capping {target_samples:,} samples to {self.config.max_samples_in_memory:,} (memory limit)")
            return self.config.max_samples_in_memory

        return target_samples

    def suggest_migration_strategy(self):
        """Suggest strategy for migrating to more powerful system"""
        print("\n" + "="*80)
        print(" SYSTEM MIGRATION GUIDE")
        print("="*80)

        if self.system_profile.system_type == 'low_end':
            print("\nCurrent System: Low-end (16GB RAM)")
            print("\nWhen you migrate to a powerful system:")
            print("  1. Copy the entire G:\\THESIS folder")
            print("  2. Ensure processed_data/ and Results/ folders are included")
            print("  3. The system will automatically detect higher resources")
            print("  4. Re-run experiments with: python run_comprehensive_experiments.py")
            print("  5. Benefits of powerful system:")
            print("     - 2-3x faster execution")
            print("     - Extended edge training percentages (up to 50%)")
            print("     - Parallel dataset processing")
            print("     - Larger models and batch sizes")

            print("\nFiles to backup before migration:")
            print("  - processed_data/*.csv (preprocessed datasets)")
            print("  - Results/checkpoints/*.pkl (experiment progress)")
            print("  - Results/HAI/*.csv (results so far)")
            print("  - Results/WADI/*.csv (results so far)")

        else:
            print("\nCurrent System: Already sufficient for comprehensive experiments")
            print("No migration needed unless you want faster execution times.")

        print("="*80)


def main():
    """Test the resource manager"""
    manager = AdaptiveResourceManager()
    manager.print_system_info()

    # Test memory check for typical datasets
    print("\n" + "="*80)
    print(" DATASET MEMORY CHECKS")
    print("="*80)

    datasets = [
        ("HAI Training", 500000, 59),
        ("HAI Testing", 150000, 59),
        ("WADI Training", 800000, 103),
        ("WADI Testing", 200000, 103),
    ]

    for name, samples, features in datasets:
        can_fit, msg = manager.check_memory_for_dataset(samples, features)
        print(f"\n{name}: {samples:,} samples x {features} features")
        print(f"  {msg}")

    manager.suggest_migration_strategy()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
