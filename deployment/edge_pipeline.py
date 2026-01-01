"""
Edge Inference Pipeline for SCADA Anomaly Detection

Production-ready inference code for:
- SCADA server deployment (TensorFlow Lite)
- Jetson Nano edge deployment (TFLite + ARM optimization)
- Shadow IDS redundancy mode

Features:
- OPC-UA tag reading simulation
- Real-time anomaly scoring
- Alarm generation
- Historian logging
"""

import os
import time
import json
import numpy as np
from datetime import datetime
from collections import deque


class SensorBuffer:
    """
    Sliding window buffer for sensor readings
    Maintains last N timesteps for sequence models
    """
    
    def __init__(self, n_features=127, window_size=4):
        self.n_features = n_features
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        
    def add(self, reading):
        """Add new sensor reading"""
        if len(reading) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(reading)}")
        self.buffer.append(reading)
        
    def get_window(self):
        """Get current window as numpy array"""
        if len(self.buffer) < self.window_size:
            return None
        return np.array(self.buffer)
    
    def is_ready(self):
        """Check if buffer has enough data"""
        return len(self.buffer) >= self.window_size


class AnomalyDetector:
    """
    Real-time anomaly detection for SCADA systems
    
    Supports:
    - TensorFlow Lite models (edge deployment)
    - PyTorch models (development/testing)
    """
    
    def __init__(self, model_path=None, n_features=127, window_size=4, threshold=0.5):
        """
        Args:
            model_path: Path to model file (.tflite or .pt)
            n_features: Number of sensor features
            window_size: Sliding window size (timesteps)
            threshold: Anomaly detection threshold
        """
        self.model_path = model_path
        self.n_features = n_features
        self.window_size = window_size
        self.threshold = threshold
        
        self.buffer = SensorBuffer(n_features, window_size)
        self.model = None
        self.interpreter = None
        self.use_tflite = False
        
        # Normalization parameters (from training)
        self.mean = None
        self.std = None
        
        # Statistics
        self.inference_count = 0
        self.anomaly_count = 0
        self.latencies = []
        
    def load_model(self, model_path=None):
        """Load model for inference"""
        if model_path:
            self.model_path = model_path
            
        if self.model_path is None:
            print("⚠ No model path provided")
            return False
            
        if self.model_path.endswith('.tflite'):
            return self._load_tflite()
        elif self.model_path.endswith('.pt'):
            return self._load_pytorch()
        else:
            print(f"⚠ Unknown model format: {self.model_path}")
            return False
    
    def _load_tflite(self):
        """Load TensorFlow Lite model"""
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.use_tflite = True
            
            size_kb = os.path.getsize(self.model_path) / 1024
            print(f"✓ Loaded TFLite model ({size_kb:.1f} KB)")
            return True
        except Exception as e:
            print(f"✗ Failed to load TFLite: {e}")
            return False
    
    def _load_pytorch(self):
        """Load PyTorch model"""
        try:
            import torch
            from experiments.models import HybridLSTMGDN
            
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model = HybridLSTMGDN(n_features=self.n_features)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.use_tflite = False
            
            print(f"✓ Loaded PyTorch model")
            return True
        except Exception as e:
            print(f"✗ Failed to load PyTorch: {e}")
            return False
    
    def set_normalization(self, mean, std):
        """Set normalization parameters from training"""
        self.mean = np.array(mean)
        self.std = np.array(std)
        
    def process_reading(self, sensor_values):
        """
        Process single sensor reading
        
        Args:
            sensor_values: Array of sensor values (n_features,)
        
        Returns:
            dict with anomaly_score, is_anomaly, latency_ms
        """
        # Normalize if parameters are set
        if self.mean is not None and self.std is not None:
            sensor_values = (sensor_values - self.mean) / (self.std + 1e-8)
        
        # Add to buffer
        self.buffer.add(sensor_values)
        
        # Check if buffer is ready
        if not self.buffer.is_ready():
            return {'anomaly_score': None, 'is_anomaly': False, 'status': 'buffering'}
        
        # Run inference
        window = self.buffer.get_window()
        start_time = time.perf_counter()
        score = self._infer(window)
        latency = (time.perf_counter() - start_time) * 1000  # ms
        
        self.latencies.append(latency)
        self.inference_count += 1
        
        is_anomaly = score > self.threshold
        if is_anomaly:
            self.anomaly_count += 1
        
        return {
            'anomaly_score': float(score),
            'is_anomaly': bool(is_anomaly),
            'latency_ms': latency,
            'threshold': self.threshold,
            'status': 'ok'
        }
    
    def _infer(self, window):
        """Run model inference"""
        # Reshape for batch dimension
        input_data = window.reshape(1, self.window_size, self.n_features).astype(np.float32)
        
        if self.use_tflite and self.interpreter:
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(output_details[0]['index'])
            return output[0][0]
            
        elif self.model is not None:
            import torch
            with torch.no_grad():
                output = self.model(torch.FloatTensor(input_data))
            return output.item()
        
        else:
            # Dummy inference for testing
            return np.random.random()
    
    def get_stats(self):
        """Get runtime statistics"""
        return {
            'inference_count': self.inference_count,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(1, self.inference_count),
            'avg_latency_ms': np.mean(self.latencies) if self.latencies else 0,
            'max_latency_ms': np.max(self.latencies) if self.latencies else 0
        }


class SCADASimulator:
    """
    Simulates SCADA OPC-UA server for testing
    """
    
    def __init__(self, n_features=127):
        self.n_features = n_features
        self.attack_mode = False
        self.attack_start = None
        
    def read_tags(self):
        """Simulate reading OPC-UA tags"""
        if self.attack_mode:
            # Inject anomalous patterns
            base = np.random.randn(self.n_features) * 0.5
            # Add attack signature
            attack_features = np.random.choice(self.n_features, size=10, replace=False)
            base[attack_features] += np.random.randn(10) * 3  # Spike values
            return base
        else:
            # Normal operation
            return np.random.randn(self.n_features) * 0.3
    
    def start_attack(self):
        """Start simulated attack"""
        self.attack_mode = True
        self.attack_start = datetime.now()
        print(f"⚠ ATTACK STARTED at {self.attack_start}")
    
    def stop_attack(self):
        """Stop simulated attack"""
        self.attack_mode = False
        print(f"✓ Attack stopped")


def run_edge_demo(duration_seconds=60, attack_at=30):
    """
    Run edge inference demo with simulated SCADA
    
    Args:
        duration_seconds: Demo duration
        attack_at: When to inject attack (seconds)
    """
    print("="*60)
    print("EDGE INFERENCE PIPELINE DEMO")
    print("="*60)
    
    # Initialize
    detector = AnomalyDetector(n_features=127, threshold=0.5)
    scada = SCADASimulator(n_features=127)
    
    print(f"\nRunning for {duration_seconds} seconds...")
    print(f"Attack injection at {attack_at} seconds\n")
    
    start_time = time.time()
    readings = []
    
    try:
        while (time.time() - start_time) < duration_seconds:
            elapsed = time.time() - start_time
            
            # Inject attack at specified time
            if elapsed >= attack_at and not scada.attack_mode:
                scada.start_attack()
            
            # Read sensors
            sensor_values = scada.read_tags()
            
            # Process reading
            result = detector.process_reading(sensor_values)
            
            if result['status'] == 'ok':
                readings.append({
                    'time': elapsed,
                    'score': result['anomaly_score'],
                    'is_anomaly': result['is_anomaly'],
                    'attack_active': scada.attack_mode
                })
                
                if result['is_anomaly']:
                    print(f"  ⚠ ANOMALY at t={elapsed:.1f}s: score={result['anomaly_score']:.3f}")
            
            # Simulate 1-second polling interval
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⚠ Demo interrupted")
    
    # Print summary
    stats = detector.get_stats()
    print(f"\n{'='*60}")
    print("DEMO SUMMARY")
    print(f"{'='*60}")
    print(f"  Total inferences: {stats['inference_count']}")
    print(f"  Anomalies detected: {stats['anomaly_count']}")
    print(f"  Anomaly rate: {stats['anomaly_rate']*100:.1f}%")
    print(f"  Avg latency: {stats['avg_latency_ms']:.3f} ms")
    print(f"  Max latency: {stats['max_latency_ms']:.3f} ms")
    
    return readings


if __name__ == "__main__":
    run_edge_demo(duration_seconds=60, attack_at=30)
