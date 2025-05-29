"""
ElevenLabs-Inspired Enhanced Audio Configuration
==============================================

This configuration implements cutting-edge audio processing techniques 
inspired by ElevenLabs' approach to achieve studio-quality voice synthesis.
"""

from dataclasses import dataclass
from typing import List, Optional
from .shared_configs import BaseAudioConfig


@dataclass 
class ElevenLabsInspiredAudioConfig(BaseAudioConfig):
    """ElevenLabs-inspired audio configuration for maximum quality."""
    
    # Core Audio Settings (ElevenLabs uses higher sample rates)
    sample_rate: int = 24000  # Higher quality than standard 22050
    resample: bool = True  # Enable resampling for quality
    
    # Advanced STFT Configuration (Multi-resolution approach)
    fft_size: int = 2048  # Base FFT size
    win_length: int = 2048  # Window length
    hop_length: int = 256  # Smaller hop for better temporal resolution
    stft_pad_mode: str = "constant"  # Better padding for quality
    
    # Multi-scale Analysis (ElevenLabs technique)
    use_multi_scale_stft: bool = True
    stft_scales: List[int] = None  # Will be set in __post_init__
    
    # Enhanced Mel-Spectrogram Settings
    num_mels: int = 128  # Higher resolution than standard 80
    mel_fmin: float = 20.0  # Lower frequency for better bass response
    mel_fmax: float = 12000.0  # Higher frequency for better clarity
    mel_scale: str = "htk"  # HTK scale for better perceptual quality
    
    # Advanced Audio Normalization
    do_sound_norm: bool = True
    do_rms_norm: bool = True
    db_level: float = -20.0  # Optimal level for voice synthesis
    preemphasis: float = 0.98  # Enhanced preemphasis
    ref_level_db: int = 20
    
    # Sophisticated Silence Handling
    do_trim_silence: bool = True
    trim_db: int = 25  # Gentle trimming to preserve natural pauses
    use_adaptive_trimming: bool = True  # Context-aware trimming
    
    # High-Quality Audio Reconstruction
    power: float = 2.2  # Enhanced power for better reconstruction
    griffin_lim_iters: int = 200  # Very high iteration count
    use_lws: bool = False  # Keep Griffin-Lim for compatibility
    
    # ElevenLabs-style Dynamic Range Optimization
    do_dynamic_range_compression: bool = True
    compression_ratio: float = 3.0
    compression_threshold: float = 0.8
    
    # Advanced Spectral Processing
    spec_gain: int = 25  # Higher gain for better detail
    min_level_db: int = -120  # Extended dynamic range
    symmetric_norm: bool = True
    max_norm: float = 5.0  # Slightly higher normalization
    clip_norm: bool = True
    signal_norm: bool = True
    
    # Formant Enhancement (ElevenLabs technique)
    use_formant_enhancement: bool = True
    formant_frequencies: List[float] = None  # Will be set in __post_init__
    formant_enhancement_factor: float = 0.15
    
    # Harmonic Enhancement
    use_harmonic_enhancement: bool = True
    harmonic_enhancement_factor: float = 0.1
    
    # Breath and Natural Sound Enhancement
    use_breath_enhancement: bool = True
    breath_enhancement_factor: float = 0.05
    preserve_natural_pauses: bool = True
    
    # Advanced Quality Metrics
    use_quality_monitoring: bool = True
    quality_assessment_interval: int = 100  # Steps between assessments
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.stft_scales is None:
            self.stft_scales = [512, 1024, 2048, 4096]
            
        if self.formant_frequencies is None:
            # Standard formant frequencies for vowel enhancement
            self.formant_frequencies = [500.0, 1500.0, 2500.0, 3500.0]


@dataclass
class ElevenLabsInspiredTrainingConfig:
    """Training configuration optimized for ElevenLabs-style quality."""
    
    # Learning Rate Optimization
    base_lr: float = 1e-6  # Very low for stability
    min_lr: float = 1e-8
    max_lr: float = 5e-6
    warmup_steps: int = 2000
    
    # Batch Size Optimization for Quality
    base_batch_size: int = 4  # Small batches for maximum quality
    gradient_accumulation_steps: int = 8  # Higher accumulation
    
    # Loss Function Weights (ElevenLabs-inspired)
    mel_loss_weight: float = 1.0
    stft_loss_weight: float = 0.5
    perceptual_loss_weight: float = 0.2
    prosody_loss_weight: float = 0.1
    discriminator_loss_weight: float = 0.05
    
    # Advanced Regularization
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 0.5
    label_smoothing: float = 0.1
    
    # Quality-Aware Training
    quality_threshold: float = 0.85
    quality_based_lr_scaling: bool = True
    early_stopping_patience: int = 10
    
    # Data Augmentation (ElevenLabs-style)
    use_pitch_augmentation: bool = True
    pitch_augmentation_range: float = 0.1  # ±10%
    use_speed_augmentation: bool = True
    speed_augmentation_range: float = 0.05  # ±5%
    use_noise_augmentation: bool = True
    noise_augmentation_factor: float = 0.02
    
    # Advanced Monitoring
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Model Architecture Optimizations
    use_attention_dropout: bool = True
    attention_dropout_rate: float = 0.1
    use_layer_norm: bool = True
    use_residual_connections: bool = True


def create_elevenlabs_audio_config() -> ElevenLabsInspiredAudioConfig:
    """Create an optimized ElevenLabs-inspired audio configuration."""
    return ElevenLabsInspiredAudioConfig()


def create_elevenlabs_training_config() -> ElevenLabsInspiredTrainingConfig:
    """Create an optimized ElevenLabs-inspired training configuration."""
    return ElevenLabsInspiredTrainingConfig()


def get_elevenlabs_optimization_report() -> dict:
    """Get a report of all ElevenLabs-inspired optimizations."""
    return {
        "audio_optimizations": [
            "24kHz sample rate for higher quality",
            "Multi-scale STFT analysis (512, 1024, 2048, 4096)",
            "128 mel channels for enhanced resolution", 
            "Adaptive silence trimming",
            "Formant frequency enhancement",
            "Harmonic enhancement for natural sound",
            "Breath sound preservation",
            "Dynamic range compression",
            "Advanced spectral processing"
        ],
        "training_optimizations": [
            "Ultra-low learning rate (1e-6) for stability",
            "Small batch sizes (4) for maximum quality",
            "High gradient accumulation (8x)",
            "Multi-component loss function",
            "Quality-aware learning rate scaling",
            "Advanced data augmentation",
            "Perceptual loss integration",
            "Real-time quality monitoring"
        ],
        "quality_features": [
            "Studio-quality 24kHz output",
            "Natural breathing preservation",
            "Enhanced vowel clarity via formants",
            "Reduced artifacts via spectral analysis",
            "Improved prosody modeling",
            "Context-aware audio processing",
            "Real-time quality assessment",
            "Adaptive optimization based on quality metrics"
        ]
    }
