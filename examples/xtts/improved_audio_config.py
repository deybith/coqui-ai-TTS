"""
Improved audio configuration for better TTS quality.
This configuration addresses common issues like:
- Audio cutting off (missing words)
- Robotic/interference sounds
- Wrong language pronunciation
"""

from dataclasses import dataclass
from xtts.shared_configs import BaseAudioConfig


@dataclass
class ImprovedAudioConfig(BaseAudioConfig):
    """Improved audio configuration for better TTS quality."""
    
    # STFT parameters - optimized for better quality
    fft_size: int = 2048  # Increased for better frequency resolution
    win_length: int = 2048  # Matched to fft_size
    hop_length: int = 512  # Increased for better temporal resolution
    stft_pad_mode: str = "reflect"
    
    # Audio processing parameters
    sample_rate: int = 22050
    resample: bool = False
    preemphasis: float = 0.97  # Slightly increased for better high-frequency emphasis
    ref_level_db: int = 20
    
    # Sound normalization - improved for consistency
    do_sound_norm: bool = True  # Enable for better volume consistency
    log_func: str = "np.log10"
    
    # Silence trimming - less aggressive to avoid cutting words
    do_trim_silence: bool = True
    trim_db: int = 30  # Reduced from 45 to be less aggressive
    
    # RMS volume normalization - enable for better consistency
    do_rms_norm: bool = True  # Enable RMS normalization
    db_level: float = -25.0  # Set appropriate level for normalization
    
    # Griffin-Lim parameters - improved for less robotic sound
    power: float = 2.0  # Increased for better reconstruction
    griffin_lim_iters: int = 100  # Increased from 60 for better quality
    
    # Mel-spectrogram parameters - optimized
    num_mels: int = 100  # Increased from 80 for better resolution
    mel_fmin: float = 50.0  # Better for most voices (was 0.0)
    mel_fmax: float = 8000.0  # Set explicit upper limit
    spec_gain: int = 20
    do_amp_to_db_linear: bool = True
    do_amp_to_db_mel: bool = True
    
    # F0 parameters - improved for better pitch tracking
    pitch_fmax: float = 800.0  # Increased from 640
    pitch_fmin: float = 50.0  # Increased from 1.0 for better pitch detection
    
    # Normalization parameters - optimized
    signal_norm: bool = True
    min_level_db: int = -120  # Reduced from -100 for better dynamic range
    symmetric_norm: bool = True
    max_norm: float = 4.0
    clip_norm: bool = True
    stats_path: str = None


@dataclass 
class ImprovedXttsAudioConfig:
    """XTTS-specific audio configuration improvements."""
    
    # Audio sampling rates
    sample_rate: int = 22050
    dvae_sample_rate: int = 22050
    output_sample_rate: int = 24000  # Keep output rate higher for better quality
    
    # Additional parameters for XTTS
    enable_redisual_residual: bool = True  # If available in your version
    use_deterministic_generation: bool = False  # Allow some randomness for naturalness


def get_improved_audio_config():
    """Return the improved audio configuration."""
    return ImprovedAudioConfig()


def get_improved_xtts_audio_config():
    """Return the improved XTTS audio configuration.""" 
    return ImprovedXttsAudioConfig()
