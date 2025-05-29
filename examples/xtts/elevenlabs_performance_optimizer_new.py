"""
ElevenLabs Performance Optimization Module
=========================================

Performance optimizations for the ElevenLabs-inspired training pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import gc
import time


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
    
    # Adaptive Training
    adaptive_batch_size: bool = True
    min_batch_size: int = 4
    max_batch_size: int = 32
    memory_threshold_mb: float = 1000.0


class ElevenLabsPerformanceOptimizer:
    """Performance optimizer for ElevenLabs-inspired training."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_tracker = MemoryTracker()
        
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
            if pred_audio.numel() > 0 and target_audio.numel() > 0:
                losses['perceptual'] = self._compute_lightweight_perceptual_loss(pred_audio, target_audio)
        else:
            # Late training: full loss computation
            losses['reconstruction'] = F.l1_loss(pred_audio, target_audio)
            if pred_audio.numel() > 0 and target_audio.numel() > 0:
                losses['perceptual'] = self._compute_lightweight_perceptual_loss(pred_audio, target_audio)
                losses['spectral'] = self._compute_spectral_loss(pred_audio, target_audio)
        
        return losses
    
    def _compute_lightweight_perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute a lightweight version of perceptual loss."""
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device if pred.numel() > 0 else target.device)
        
        # Simple magnitude comparison
        pred_mag = pred.abs().mean()
        target_mag = target.abs().mean()
        return F.l1_loss(pred_mag, target_mag)
    
    def _compute_spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss efficiently."""
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device if pred.numel() > 0 else target.device)
        
        try:
            # Efficient STFT computation
            if pred.dim() > 1:
                pred = pred.squeeze(1)
            if target.dim() > 1:
                target = target.squeeze(1)
                
            pred_stft = torch.stft(pred, n_fft=1024, hop_length=256, 
                                  return_complex=True, normalized=True)
            target_stft = torch.stft(target, n_fft=1024, hop_length=256, 
                                    return_complex=True, normalized=True)
            
            return F.l1_loss(pred_stft.abs(), target_stft.abs())
        except:
            return torch.tensor(0.0, device=pred.device)
    
    def memory_cleanup(self):
        """Perform memory cleanup."""
        if self.config.enable_memory_optimization:
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
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
    
    # Set optimal number of threads
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)  # Often optimal for training
    
    print("✓ Training environment optimized for performance")


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
