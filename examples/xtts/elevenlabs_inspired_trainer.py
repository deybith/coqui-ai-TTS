"""
ElevenLabs-Inspired Ultra Enhanced Trainer with Performance Optimizations
========================================================================

This trainer integrates cutting-edge audio quality techniques inspired by 
ElevenLabs' approach to achieve studio-quality voice synthesis with 
comprehensive performance optimizations.
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time

from trainer import Trainer
from .elevenlabs_quality_enhancement import (
    ElevenLabsConfig,
    ElevenLabsQualityLoss,
    ElevenLabsAudioProcessor,
    get_elevenlabs_config,
    create_elevenlabs_loss_function,
    create_elevenlabs_processor
)
from .elevenlabs_performance_optimizer import (
    ElevenLabsPerformanceOptimizer,
    PerformanceConfig,
    create_performance_optimizer,
    optimize_training_environment
)


class ElevenLabsInspiredTrainer(Trainer):
    """Ultra-enhanced trainer with ElevenLabs-inspired audio quality techniques and performance optimizations."""
    
    def __init__(self, *args, **kwargs):
        # Extract performance configuration if provided
        performance_config = kwargs.pop('performance_config', None)
        
        super().__init__(*args, **kwargs)
        
        # Initialize ElevenLabs-inspired components
        self.elevenlabs_config = get_elevenlabs_config()
        self.elevenlabs_loss = create_elevenlabs_loss_function(self.elevenlabs_config)
        self.audio_processor = create_elevenlabs_processor(self.elevenlabs_config)
        
        # Initialize performance optimizer
        self.performance_optimizer = create_performance_optimizer(performance_config)
        
        # Optimize training environment
        optimize_training_environment()
        
        # Quality monitoring
        self.quality_history = []
        self.best_quality_score = 0.0
        self.quality_improvement_threshold = 0.01
        
        # Training optimization
        self.use_gradient_penalty = True
        self.gradient_penalty_weight = 10.0
        
        # Audio quality metrics
        self.quality_metrics = {
            'spectral_quality': [],
            'perceptual_quality': [],
            'prosody_quality': [],
            'overall_quality': []
        }
        
        # Performance metrics
        self.training_times = []
        self.memory_usage = []
        
        print("🎵 ElevenLabs-Inspired Trainer Initialized with Performance Optimizations")
        print("✨ Features: Multi-scale loss, Prosody modeling, Quality assessment, Memory optimization")
        print("🚀 Performance: Adaptive batching, Progressive training, Smart caching")
    
    def train_step(self, batch: Dict[str, Any], batch_n_steps: int, step: int, loader_start_time: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Enhanced training step with ElevenLabs-inspired quality optimization and performance improvements."""
        
        # Start timing for performance monitoring
        step_start_time = time.time()
        
        # Optimize batch processing
        batch = self.performance_optimizer.optimize_batch_processing(batch)
        
        # Get original outputs and loss from base trainer
        outputs, loss_dict = super().train_step(batch, batch_n_steps, step, loader_start_time)
        
        if outputs is None or loss_dict is None:
            return None, None
        
        if self.model.training:
            # Apply ElevenLabs-inspired enhancements with adaptive loss computation
            enhanced_losses = self.performance_optimizer.adaptive_loss_computation(
                outputs.get('wav', torch.zeros(1)), 
                batch.get('wav', torch.zeros(1)), 
                step
            )
            
            # Also compute full ElevenLabs losses
            full_enhanced_losses = self._compute_elevenlabs_losses(batch, outputs)
            
            # Merge all losses
            for key, value in enhanced_losses.items():
                if key in loss_dict:
                    loss_dict[key] = loss_dict[key] + value
                else:
                    loss_dict[key] = value
                    
            for key, value in full_enhanced_losses.items():
                if key in loss_dict:
                    loss_dict[key] = loss_dict[key] + value * 0.5  # Scale to avoid over-weighting
                else:
                    loss_dict[key] = value * 0.5
            
            # Quality-aware learning rate adjustment
            if self.elevenlabs_config.use_real_time_quality_assessment:
                self._adjust_learning_rate_by_quality(loss_dict)
        
        # Track performance metrics
        step_time = time.time() - step_start_time
        self.training_times.append(step_time)
        
        # Memory cleanup if needed
        if step % 100 == 0:
            self.performance_optimizer.memory_cleanup()
            
        # Track memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
            self.memory_usage.append(memory_usage)
        
        return outputs, loss_dict
    
    def _compute_elevenlabs_losses(self, batch: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute ElevenLabs-inspired quality losses."""
        enhanced_losses = {}
        
        try:
            # Extract audio tensors
            if 'wav' in outputs and 'wav' in batch:
                pred_audio = outputs['wav']
                target_audio = batch['wav']
                
                # Ensure same length
                min_length = min(pred_audio.shape[-1], target_audio.shape[-1])
                pred_audio = pred_audio[..., :min_length]
                target_audio = target_audio[..., :min_length]
                
                # Compute ElevenLabs-inspired losses
                quality_losses = self.elevenlabs_loss(pred_audio, target_audio)
                
                # Scale losses appropriately
                for loss_name, loss_value in quality_losses.items():
                    enhanced_losses[f'elevenlabs_{loss_name}'] = loss_value
                    
                # Compute quality score
                quality_score = self._compute_quality_score(quality_losses)
                enhanced_losses['quality_score'] = quality_score
                self.quality_history.append(quality_score.item())
                
        except Exception as e:
            print(f"⚠️  Warning: Could not compute ElevenLabs losses: {e}")
            # Return empty dict if computation fails
            enhanced_losses = {}
            
        return enhanced_losses
    
    def _compute_quality_score(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute overall quality score from losses."""
        # Lower loss = higher quality
        # Convert losses to quality scores (0-1, higher is better)
        
        quality_components = []
        
        if 'mel_loss' in losses:
            mel_quality = torch.exp(-losses['mel_loss'] * 10)  # Convert loss to quality
            quality_components.append(mel_quality)
            
        if 'stft_spectral_convergence' in losses:
            spectral_quality = torch.exp(-losses['stft_spectral_convergence'] * 5)
            quality_components.append(spectral_quality)
            
        if 'energy_loss' in losses:
            prosody_quality = torch.exp(-losses['energy_loss'] * 20)
            quality_components.append(prosody_quality)
            
        if quality_components:
            overall_quality = torch.stack(quality_components).mean()
        else:
            overall_quality = torch.tensor(0.5, device=self.device)  # Default neutral quality
            
        return overall_quality
    
    def _adjust_learning_rate_by_quality(self, loss_dict: Dict[str, Any]) -> None:
        """Dynamically adjust learning rate based on quality metrics."""
        if len(self.quality_history) < 10:  # Need some history
            return
            
        # Check if quality is improving
        recent_quality = np.mean(self.quality_history[-5:])
        older_quality = np.mean(self.quality_history[-10:-5])
        
        if recent_quality > older_quality + self.quality_improvement_threshold:
            # Quality improving - can increase LR slightly
            self._scale_learning_rate(1.02)
        elif recent_quality < older_quality - self.quality_improvement_threshold:
            # Quality degrading - reduce LR
            self._scale_learning_rate(0.98)
    
    def _scale_learning_rate(self, factor: float) -> None:
        """Scale learning rate by a factor."""
        if isinstance(self.optimizer, list):
            for optimizer in self.optimizer:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= factor
    
    def eval_step(self, batch: Dict[str, Any], step: int) -> Tuple[Any, Dict[str, Any]]:
        """Enhanced evaluation step with quality assessment."""
        outputs, loss_dict = super().eval_step(batch, step)
        
        # Add quality assessment for evaluation
        if outputs is not None:
            quality_metrics = self._assess_audio_quality(batch, outputs)
            loss_dict.update(quality_metrics)
            
        return outputs, loss_dict
    
    def _assess_audio_quality(self, batch: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Assess audio quality using multiple metrics."""
        quality_metrics = {}
        
        try:
            if 'wav' in outputs and 'wav' in batch:
                pred_audio = outputs['wav']
                target_audio = batch['wav']
                
                # Ensure same length
                min_length = min(pred_audio.shape[-1], target_audio.shape[-1])
                pred_audio = pred_audio[..., :min_length]
                target_audio = target_audio[..., :min_length]
                
                # Spectral quality
                spectral_distance = self._compute_spectral_distance(pred_audio, target_audio)
                quality_metrics['eval_spectral_quality'] = spectral_distance
                
                # Perceptual quality (simplified PESQ-like metric)
                perceptual_quality = self._compute_perceptual_quality(pred_audio, target_audio)
                quality_metrics['eval_perceptual_quality'] = perceptual_quality
                
                # Overall quality score
                overall_quality = (spectral_distance + perceptual_quality) / 2
                quality_metrics['eval_overall_quality'] = overall_quality
                
                # Update quality metrics history
                self.quality_metrics['spectral_quality'].append(spectral_distance.item())
                self.quality_metrics['perceptual_quality'].append(perceptual_quality.item())
                self.quality_metrics['overall_quality'].append(overall_quality.item())
                
        except Exception as e:
            print(f"⚠️  Warning: Could not assess audio quality: {e}")
            
        return quality_metrics
    
    def _compute_spectral_distance(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """Compute spectral distance between predicted and target audio."""
        # Compute mel spectrograms
        n_mels = 128
        pred_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.elevenlabs_config.target_sample_rate,
            n_mels=n_mels,
            n_fft=self.elevenlabs_config.n_fft,
            hop_length=self.elevenlabs_config.hop_length
        )(pred_audio)
        
        target_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.elevenlabs_config.target_sample_rate,
            n_mels=n_mels,
            n_fft=self.elevenlabs_config.n_fft,
            hop_length=self.elevenlabs_config.hop_length
        )(target_audio)
        
        # Log mel spectrograms
        pred_log_mel = torch.log(pred_mel + 1e-8)
        target_log_mel = torch.log(target_mel + 1e-8)
        
        # Compute spectral distance (higher is better)
        distance = 1.0 / (1.0 + F.mse_loss(pred_log_mel, target_log_mel))
        return distance
    
    def _compute_perceptual_quality(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """Compute perceptual quality score."""
        # Multi-scale STFT comparison
        quality_scores = []
        
        for n_fft in [512, 1024, 2048]:
            hop_length = n_fft // 4
            
            pred_stft = torch.stft(pred_audio, n_fft=n_fft, hop_length=hop_length, 
                                 return_complex=True, normalized=True)
            target_stft = torch.stft(target_audio, n_fft=n_fft, hop_length=hop_length, 
                                   return_complex=True, normalized=True)
            
            # Magnitude comparison
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            
            # Perceptual quality (inverse of normalized MSE)
            mse = F.mse_loss(pred_mag, target_mag)
            quality = 1.0 / (1.0 + mse)
            quality_scores.append(quality)
            
        return torch.stack(quality_scores).mean()
    
    def on_epoch_end(self, trainer) -> None:
        """Enhanced epoch end with quality reporting."""
        super().on_epoch_end(trainer)
        
        # Report quality metrics
        if self.quality_metrics['overall_quality']:
            avg_quality = np.mean(self.quality_metrics['overall_quality'][-10:])  # Last 10 evaluations
            
            if avg_quality > self.best_quality_score:
                self.best_quality_score = avg_quality
                print(f"🎵 New best audio quality: {avg_quality:.3f}")
                
                # Save best quality model
                if hasattr(self, 'save_best_model'):
                    self.save_best_model()
        
        # Print quality summary
        if len(self.quality_history) > 0:
            recent_training_quality = np.mean(self.quality_history[-50:])  # Last 50 training steps
            print(f"📊 Recent training quality: {recent_training_quality:.3f}")
            
        # Clear old history to prevent memory issues
        if len(self.quality_history) > 1000:
            self.quality_history = self.quality_history[-500:]
            
        for metric_name, values in self.quality_metrics.items():
            if len(values) > 100:
                self.quality_metrics[metric_name] = values[-50:]
    
    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ElevenLabs-inspired preprocessing to batch."""
        # Apply audio enhancement if needed
        if 'wav' in batch and self.elevenlabs_config.use_harmonic_enhancement:
            try:
                # Convert to numpy for processing
                audio_tensor = batch['wav']
                batch_size = audio_tensor.shape[0]
                
                enhanced_audio = []
                for i in range(batch_size):
                    audio_np = audio_tensor[i].cpu().numpy()
                    # Apply enhancement
                    enhanced = self.audio_processor.enhance_audio_quality(audio_np)
                    enhanced_audio.append(enhanced)
                
                # Convert back to tensor
                enhanced_tensor = torch.tensor(enhanced_audio, dtype=audio_tensor.dtype, device=audio_tensor.device)
                batch['wav'] = enhanced_tensor
                
            except Exception as e:
                print(f"⚠️  Warning: Could not apply audio enhancement: {e}")
                
        return batch
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality and performance report."""
        report = {
            # Quality metrics
            'best_quality_score': self.best_quality_score,
            'current_training_quality': np.mean(self.quality_history[-10:]) if self.quality_history else 0.0,
            'quality_trend': 'improving' if len(self.quality_history) > 20 and 
                           np.mean(self.quality_history[-10:]) > np.mean(self.quality_history[-20:-10]) else 'stable',
            'total_evaluations': len(self.quality_metrics['overall_quality']),
            'average_spectral_quality': np.mean(self.quality_metrics['spectral_quality']) if self.quality_metrics['spectral_quality'] else 0.0,
            'average_perceptual_quality': np.mean(self.quality_metrics['perceptual_quality']) if self.quality_metrics['perceptual_quality'] else 0.0,
            
            # Performance metrics
            'average_step_time': np.mean(self.training_times[-100:]) if self.training_times else 0.0,
            'current_memory_usage_mb': self.memory_usage[-1] if self.memory_usage else 0.0,
            'peak_memory_usage_mb': max(self.memory_usage) if self.memory_usage else 0.0,
            'steps_per_second': 1.0 / np.mean(self.training_times[-100:]) if self.training_times else 0.0,
            
            # Training efficiency
            'memory_efficiency': 'good' if (self.memory_usage[-1] if self.memory_usage else 0) < 8000 else 'high',
            'speed_efficiency': 'good' if (np.mean(self.training_times[-10:]) if self.training_times else 1.0) < 2.0 else 'slow',
        }
        
        return report
    
    def print_comprehensive_status(self):
        """Print comprehensive training status including quality and performance metrics."""
        report = self.get_quality_report()
        
        print("\n" + "="*60)
        print("🎵 ElevenLabs-Inspired Training Status Report")
        print("="*60)
        
        # Quality metrics
        print(f"🎯 Audio Quality:")
        print(f"   Best Quality Score: {report['best_quality_score']:.3f}")
        print(f"   Current Quality: {report['current_training_quality']:.3f}")
        print(f"   Quality Trend: {report['quality_trend']}")
        print(f"   Spectral Quality: {report['average_spectral_quality']:.3f}")
        print(f"   Perceptual Quality: {report['average_perceptual_quality']:.3f}")
        
        # Performance metrics
        print(f"\n⚡ Performance:")
        print(f"   Steps/Second: {report['steps_per_second']:.2f}")
        print(f"   Avg Step Time: {report['average_step_time']:.3f}s")
        print(f"   Current Memory: {report['current_memory_usage_mb']:.1f} MB")
        print(f"   Peak Memory: {report['peak_memory_usage_mb']:.1f} MB")
        print(f"   Memory Efficiency: {report['memory_efficiency']}")
        print(f"   Speed Efficiency: {report['speed_efficiency']}")
        
        print("="*60 + "\n")
    
    def optimize_model_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference using performance optimizations."""
        return self.performance_optimizer.optimize_model(model)
