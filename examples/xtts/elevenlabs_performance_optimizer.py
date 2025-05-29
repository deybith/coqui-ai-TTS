"""
ElevenLabs Performance Optimization Module
=========================================

This module provides performance optimizations for the ElevenLabs-inspired
training pipeline to ensure efficient training while maintaining audio quality.

Key Features:
- Memory optimization
- Computational efficiency improvements
- Adaptive batch processing
- Smart caching mechanisms
- Progressive training strategies
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import gc
from functools import lru_cache


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    
    # Memory Management
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    gradient_checkpointing: bool = True
    
    # Computation Optimization
    use_mixed_precision: bool = True
    enable_torch_compile: bool = True
    use_flash_attention: bool = True
    
    # Adaptive Training
    adaptive_batch_size: bool = True
    min_batch_size: int = 4
    max_batch_size: int = 32
    memory_threshold_mb: float = 1000.0
    
    # Caching
    enable_spectrogram_cache: bool = True
    cache_size_limit: int = 1000
    enable_feature_cache: bool = True
    
    # Progressive Training
    enable_progressive_training: bool = True
    start_resolution: int = 128
    target_resolution: int = 512
    resolution_steps: int = 4


class ElevenLabsPerformanceOptimizer:
    """Performance optimizer for ElevenLabs-inspired training."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.spectrogram_cache = {}
        self.feature_cache = {}
        self.memory_tracker = MemoryTracker()
        self.adaptive_batch_manager = AdaptiveBatchManager(config)
        
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply model-level optimizations."""
        
        # Enable gradient checkpointing to save memory
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        
        # Compile model for better performance (PyTorch 2.0+)
        if self.config.enable_torch_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("✓ Model compiled for better performance")
            except Exception as e:
                print(f"Warning: Model compilation failed: {e}")
        
        return model
    
    def optimize_batch_processing(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize batch processing for efficiency."""
        
        # Memory-efficient tensor operations
        optimized_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Use memory-efficient data types where appropriate
                if value.dtype == torch.float64:
                    optimized_batch[key] = value.to(torch.float32)
                elif key in ['mel', 'linear'] and self.config.use_mixed_precision:
                    optimized_batch[key] = value.half()
                else:
                    optimized_batch[key] = value
            else:
                optimized_batch[key] = value
        
        return optimized_batch
    
    @lru_cache(maxsize=1000)
    def cached_spectrogram_computation(self, audio_hash: str, params_hash: str) -> torch.Tensor:
        """Cached spectrogram computation to avoid redundant calculations."""
        # This would typically compute and cache spectrograms
        # Implementation depends on your specific spectrogram computation
        pass
    
    def adaptive_loss_computation(self, pred_audio: torch.Tensor, 
                                target_audio: torch.Tensor, 
                                step: int) -> Dict[str, torch.Tensor]:
        """Compute losses adaptively based on training progress."""
        
        losses = {}
        
        # Progressive loss complexity
        if step < 1000:
            # Early training: focus on basic reconstruction
            losses['reconstruction'] = F.l1_loss(pred_audio, target_audio)
        elif step < 5000:
            # Mid training: add perceptual loss
            losses['reconstruction'] = F.l1_loss(pred_audio, target_audio)
            losses['perceptual'] = self._compute_lightweight_perceptual_loss(pred_audio, target_audio)
        else:
            # Late training: full loss computation
            losses['reconstruction'] = F.l1_loss(pred_audio, target_audio)
            losses['perceptual'] = self._compute_full_perceptual_loss(pred_audio, target_audio)
            losses['spectral'] = self._compute_spectral_loss(pred_audio, target_audio)
        
        return losses
    
    def _compute_lightweight_perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute a lightweight version of perceptual loss."""
        # Simple mel-spectrogram loss
        pred_mel = self._to_mel_spectrogram(pred)
        target_mel = self._to_mel_spectrogram(target)
        return F.l1_loss(pred_mel, target_mel)
    
    def _compute_full_perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute full perceptual loss with all features."""
        # Multi-scale mel-spectrogram loss
        total_loss = 0.0
        
        for hop_length in [256, 512, 1024]:
            pred_mel = self._to_mel_spectrogram(pred, hop_length=hop_length)
            target_mel = self._to_mel_spectrogram(target, hop_length=hop_length)
            total_loss += F.l1_loss(pred_mel, target_mel)
        
        return total_loss / 3.0
    
    def _compute_spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss efficiently."""
        # Efficient STFT computation
        pred_stft = torch.stft(pred.squeeze(1), n_fft=1024, hop_length=256, 
                              return_complex=True, normalized=True)
        target_stft = torch.stft(target.squeeze(1), n_fft=1024, hop_length=256, 
                                return_complex=True, normalized=True)
        
        return F.l1_loss(pred_stft.abs(), target_stft.abs())
    
    @torch.no_grad()
    def _to_mel_spectrogram(self, audio: torch.Tensor, hop_length: int = 256) -> torch.Tensor:
        """Convert audio to mel spectrogram efficiently."""
        # This is a simplified implementation
        # In practice, you'd use a proper mel-spectrogram transform
        stft = torch.stft(audio.squeeze(1), n_fft=1024, hop_length=hop_length, 
                         return_complex=True, normalized=True)
        return stft.abs()
    
    def memory_cleanup(self):
        """Perform memory cleanup."""
        if self.config.enable_memory_optimization:
            # Clear caches if they get too large
            if len(self.spectrogram_cache) > self.config.cache_size_limit:
                self.spectrogram_cache.clear()
            
            if len(self.feature_cache) > self.config.cache_size_limit:
                self.feature_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class MemoryTracker:
    """Track memory usage during training."""
    
    def __init__(self):
        self.peak_memory = 0.0
        self.current_memory = 0.0
        
    def update(self):
        """Update memory tracking."""
        if torch.cuda.is_available():
            self.current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.peak_memory = max(self.peak_memory, self.current_memory)
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        self.update()
        return {
            'current_memory_gb': self.current_memory,
            'peak_memory_gb': self.peak_memory
        }


class AdaptiveBatchManager:
    """Manage batch size adaptively based on memory usage."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.current_batch_size = config.min_batch_size
        self.memory_history = []
        
    def update_batch_size(self, memory_usage_mb: float) -> int:
        """Update batch size based on memory usage."""
        if not self.config.adaptive_batch_size:
            return self.current_batch_size
        
        self.memory_history.append(memory_usage_mb)
        
        # Keep only recent history
        if len(self.memory_history) > 10:
            self.memory_history = self.memory_history[-10:]
        
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        
        # Adjust batch size based on memory usage
        if avg_memory > self.config.memory_threshold_mb and self.current_batch_size > self.config.min_batch_size:
            self.current_batch_size = max(self.config.min_batch_size, self.current_batch_size - 2)
        elif avg_memory < self.config.memory_threshold_mb * 0.7 and self.current_batch_size < self.config.max_batch_size:
            self.current_batch_size = min(self.config.max_batch_size, self.current_batch_size + 2)
        
        return self.current_batch_size


class ProgressiveTrainingManager:
    """Manage progressive training strategy."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.current_resolution = config.start_resolution
        
    def should_increase_resolution(self, step: int, total_steps: int) -> bool:
        """Determine if resolution should be increased."""
        if not self.config.enable_progressive_training:
            return False
        
        progress = step / total_steps
        resolution_progress = (self.current_resolution - self.config.start_resolution) / \
                            (self.config.target_resolution - self.config.start_resolution)
        
        return progress > resolution_progress + (1.0 / self.config.resolution_steps)
    
    def get_current_resolution(self) -> int:
        """Get current resolution for training."""
        return self.current_resolution
    
    def increase_resolution(self):
        """Increase the current resolution."""
        step_size = (self.config.target_resolution - self.config.start_resolution) // self.config.resolution_steps
        self.current_resolution = min(self.config.target_resolution, 
                                    self.current_resolution + step_size)


def create_performance_optimizer(config: Optional[PerformanceConfig] = None) -> ElevenLabsPerformanceOptimizer:
    """Create a performance optimizer with default or custom configuration."""
    if config is None:
        config = PerformanceConfig()
    
    return ElevenLabsPerformanceOptimizer(config)


def optimize_training_environment():
    """Optimize the training environment for best performance."""
    
    # Set optimal PyTorch settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable optimized attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    # Set optimal number of threads
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)  # Often optimal for training
    
    print("✓ Training environment optimized for performance")


# Example usage configuration
def get_optimal_config_for_gpu_memory(gpu_memory_gb: float) -> PerformanceConfig:
    """Get optimal configuration based on available GPU memory."""
    
    if gpu_memory_gb >= 24:  # High-end GPU
        return PerformanceConfig(
            max_memory_usage_gb=20.0,
            max_batch_size=32,
            enable_torch_compile=True,
            gradient_checkpointing=False,  # Don't need it with lots of memory
        )
    elif gpu_memory_gb >= 12:  # Mid-range GPU
        return PerformanceConfig(
            max_memory_usage_gb=10.0,
            max_batch_size=16,
            enable_torch_compile=True,
            gradient_checkpointing=True,
        )
    else:  # Low-end GPU
        return PerformanceConfig(
            max_memory_usage_gb=6.0,
            max_batch_size=8,
            enable_torch_compile=False,  # May cause issues on older GPUs
            gradient_checkpointing=True,
            use_mixed_precision=True,
        )
