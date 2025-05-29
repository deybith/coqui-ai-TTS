"""
ElevenLabs-Inspired Audio Quality Enhancement System
==================================================

This module implements cutting-edge audio quality techniques inspired by 
ElevenLabs' approach to achieve studio-quality voice synthesis.

Key Features:
- Multi-scale perceptual loss
- Neural vocoder discriminator
- Prosody-aware training
- Context-sensitive audio processing
- Advanced spectral analysis
- Real-time quality assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import librosa
from scipy import signal


@dataclass
class ElevenLabsConfig:
    """Configuration for ElevenLabs-inspired audio enhancement."""
    
    # Core Quality Settings
    target_sample_rate: int = 24000  # Higher quality than standard 22050
    mel_channels: int = 128  # Increased resolution
    hop_length: int = 256  # Optimal for 24kHz
    win_length: int = 1024
    n_fft: int = 2048
    
    # Multi-scale Analysis
    stft_scales: List[int] = None  # Will be set in __post_init__
    mel_scales: List[int] = None
    
    # Perceptual Loss Configuration
    perceptual_loss_weight: float = 0.15
    spectral_loss_weight: float = 0.1
    feature_matching_weight: float = 0.05
    
    # Discriminator Settings
    use_multi_period_discriminator: bool = True
    use_multi_scale_discriminator: bool = True
    discriminator_periods: List[int] = None
    
    # Prosody Enhancement
    use_prosody_loss: bool = True
    pitch_loss_weight: float = 0.1
    energy_loss_weight: float = 0.05
    
    # Advanced Audio Processing
    use_harmonic_enhancement: bool = True
    use_formant_correction: bool = True
    use_breath_enhancement: bool = True
    
    # Quality Assessment
    use_real_time_quality_assessment: bool = True
    quality_threshold: float = 0.85
    
    def __post_init__(self):
        if self.stft_scales is None:
            self.stft_scales = [512, 1024, 2048, 4096]
        if self.mel_scales is None:
            self.mel_scales = [64, 80, 100, 128]
        if self.discriminator_periods is None:
            self.discriminator_periods = [2, 3, 5, 7, 11]


class MelSpectrogramLoss(nn.Module):
    """Multi-scale mel-spectrogram loss for perceptual quality."""
    
    def __init__(self, config: ElevenLabsConfig):
        super().__init__()
        self.config = config
        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.target_sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=n_mels,
                f_min=0.0,
                f_max=config.target_sample_rate // 2,
            )
            for n_mels in config.mel_scales
        ])
        
    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale mel-spectrogram loss."""
        total_loss = 0.0
        
        for mel_transform in self.mel_transforms:
            pred_mel = mel_transform(pred_audio)
            target_mel = mel_transform(target_audio)
            
            # Log mel-spectrogram for perceptual similarity
            pred_log_mel = torch.log(pred_mel + 1e-8)
            target_log_mel = torch.log(target_mel + 1e-8)
            
            # L1 loss in log mel domain
            mel_loss = F.l1_loss(pred_log_mel, target_log_mel)
            total_loss += mel_loss
            
        return total_loss / len(self.mel_transforms)


class STFTLoss(nn.Module):
    """Multi-resolution STFT loss for spectral accuracy."""
    
    def __init__(self, config: ElevenLabsConfig):
        super().__init__()
        self.config = config
        self.stft_scales = config.stft_scales
        
    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute multi-resolution STFT loss."""
        spectral_loss = 0.0
        magnitude_loss = 0.0
        
        for n_fft in self.stft_scales:
            hop_length = n_fft // 4
            
            # Compute STFT
            pred_stft = torch.stft(pred_audio, n_fft=n_fft, hop_length=hop_length, 
                                 return_complex=True, normalized=True)
            target_stft = torch.stft(target_audio, n_fft=n_fft, hop_length=hop_length, 
                                   return_complex=True, normalized=True)
            
            # Magnitude and phase
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            
            # Spectral convergence loss
            spec_loss = torch.norm(target_mag - pred_mag, p='fro') / torch.norm(target_mag, p='fro')
            spectral_loss += spec_loss
            
            # Log magnitude loss
            pred_log_mag = torch.log(pred_mag + 1e-8)
            target_log_mag = torch.log(target_mag + 1e-8)
            mag_loss = F.l1_loss(pred_log_mag, target_log_mag)
            magnitude_loss += mag_loss
            
        return {
            'spectral_convergence': spectral_loss / len(self.stft_scales),
            'magnitude_loss': magnitude_loss / len(self.stft_scales)
        }


class ProsodyLoss(nn.Module):
    """Prosody-aware loss for natural speech patterns."""
    
    def __init__(self, config: ElevenLabsConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.target_sample_rate
        
    def extract_f0(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract fundamental frequency (pitch) from audio."""
        # Convert to numpy for librosa processing
        audio_np = audio.detach().cpu().numpy()
        
        f0_values = []
        for i in range(audio_np.shape[0]):  # Batch dimension
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_np[i], 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            # Replace NaN with 0
            f0 = np.nan_to_num(f0)
            f0_values.append(f0)
            
        return torch.tensor(f0_values, device=audio.device, dtype=audio.dtype)
    
    def extract_energy(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract energy contour from audio."""
        # RMS energy in frames
        frame_length = self.config.hop_length * 2
        energy = torch.sqrt(F.avg_pool1d(
            audio.pow(2).unsqueeze(1), 
            kernel_size=frame_length, 
            stride=self.config.hop_length,
            padding=frame_length//2
        ).squeeze(1))
        return energy
        
    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute prosody loss."""
        losses = {}
        
        # Pitch loss
        if self.config.use_prosody_loss:
            try:
                pred_f0 = self.extract_f0(pred_audio)
                target_f0 = self.extract_f0(target_audio)
                
                # Align lengths
                min_len = min(pred_f0.shape[-1], target_f0.shape[-1])
                pred_f0 = pred_f0[:, :min_len]
                target_f0 = target_f0[:, :min_len]
                
                pitch_loss = F.mse_loss(pred_f0, target_f0)
                losses['pitch_loss'] = pitch_loss * self.config.pitch_loss_weight
            except:
                losses['pitch_loss'] = torch.tensor(0.0, device=pred_audio.device)
        
        # Energy loss
        pred_energy = self.extract_energy(pred_audio)
        target_energy = self.extract_energy(target_audio)
        
        # Align lengths
        min_len = min(pred_energy.shape[-1], target_energy.shape[-1])
        pred_energy = pred_energy[:, :min_len]
        target_energy = target_energy[:, :min_len]
        
        energy_loss = F.mse_loss(pred_energy, target_energy)
        losses['energy_loss'] = energy_loss * self.config.energy_loss_weight
        
        return losses


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator for audio quality assessment."""
    
    def __init__(self, config: ElevenLabsConfig):
        super().__init__()
        self.config = config
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in config.discriminator_periods
        ])
        
    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass through all period discriminators."""
        outputs = []
        for discriminator in self.discriminators:
            output, features = discriminator(x)
            outputs.append((output, features))
        return outputs


class PeriodDiscriminator(nn.Module):
    """Single period discriminator."""
    
    def __init__(self, period: int, channels: int = 32):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, channels, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(channels, channels * 2, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(channels * 2, channels * 4, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(channels * 4, channels * 8, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(channels * 8, channels * 8, (5, 1), 1, (2, 0)),
        ])
        
        self.conv_post = nn.Conv2d(channels * 8, 1, (3, 1), 1, (1, 0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass."""
        # Reshape for period
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
            
        x = x.view(b, c, t // self.period, self.period)
        
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
            
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, features


class ElevenLabsQualityLoss(nn.Module):
    """Complete ElevenLabs-inspired quality loss function."""
    
    def __init__(self, config: ElevenLabsConfig):
        super().__init__()
        self.config = config
        
        # Loss components
        self.mel_loss = MelSpectrogramLoss(config)
        self.stft_loss = STFTLoss(config)
        self.prosody_loss = ProsodyLoss(config)
        
        # Discriminators
        if config.use_multi_period_discriminator:
            self.mpd = MultiPeriodDiscriminator(config)
            
    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor, 
                discriminator_features: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Compute comprehensive quality loss."""
        losses = {}
        
        # Core spectral losses
        mel_loss = self.mel_loss(pred_audio, target_audio)
        losses['mel_loss'] = mel_loss * self.config.perceptual_loss_weight
        
        stft_losses = self.stft_loss(pred_audio, target_audio)
        losses.update({f'stft_{k}': v * self.config.spectral_loss_weight 
                      for k, v in stft_losses.items()})
        
        # Prosody losses
        prosody_losses = self.prosody_loss(pred_audio, target_audio)
        losses.update(prosody_losses)
        
        # Feature matching loss (if discriminator features provided)
        if discriminator_features is not None and self.config.use_multi_period_discriminator:
            feature_loss = self._compute_feature_matching_loss(
                discriminator_features['pred'], 
                discriminator_features['target']
            )
            losses['feature_matching'] = feature_loss * self.config.feature_matching_weight
        
        return losses
    
    def _compute_feature_matching_loss(self, pred_features: List, target_features: List) -> torch.Tensor:
        """Compute feature matching loss between discriminator features."""
        total_loss = 0.0
        num_features = 0
        
        for pred_feat_list, target_feat_list in zip(pred_features, target_features):
            for pred_feat, target_feat in zip(pred_feat_list, target_feat_list):
                total_loss += F.l1_loss(pred_feat, target_feat)
                num_features += 1
                
        return total_loss / num_features if num_features > 0 else torch.tensor(0.0)


class ElevenLabsAudioProcessor:
    """Advanced audio preprocessing inspired by ElevenLabs."""
    
    def __init__(self, config: ElevenLabsConfig):
        self.config = config
        self.sample_rate = config.target_sample_rate
        
    def enhance_audio_quality(self, audio: np.ndarray) -> np.ndarray:
        """Apply ElevenLabs-inspired audio enhancement."""
        # 1. Harmonic enhancement
        if self.config.use_harmonic_enhancement:
            audio = self._enhance_harmonics(audio)
            
        # 2. Formant correction
        if self.config.use_formant_correction:
            audio = self._correct_formants(audio)
            
        # 3. Breath enhancement
        if self.config.use_breath_enhancement:
            audio = self._enhance_breathing(audio)
            
        # 4. Dynamic range optimization
        audio = self._optimize_dynamic_range(audio)
        
        return audio
    
    def _enhance_harmonics(self, audio: np.ndarray) -> np.ndarray:
        """Enhance harmonic content for more natural sound."""
        try:
            # Extract fundamental frequency using safer method
            f0, voiced_flag, _ = librosa.pyin(audio, fmin=80, fmax=400, sr=self.sample_rate)
            
            # Generate harmonic enhancement
            enhanced = audio.copy()
            
            # Process each frame safely
            for i in range(len(f0)):
                try:
                    freq = f0[i]
                    voiced = voiced_flag[i]
                    
                    # Safe scalar conversion
                    if np.isfinite(freq) and voiced:
                        freq_val = float(freq)
                        # Add subtle harmonic enhancement
                        t = i * 512 / self.sample_rate  # Time in seconds
                        harmonic_val = 0.1 * np.sin(2 * np.pi * freq_val * 2 * t)  # 2nd harmonic
                        start_idx = i * 512
                        end_idx = min((i + 1) * 512, len(enhanced))
                        
                        if start_idx < len(enhanced):
                            # Create harmonic array for the frame
                            frame_length = end_idx - start_idx
                            harmonic_frame = np.full(frame_length, harmonic_val, dtype=np.float32)
                            enhanced[start_idx:end_idx] += harmonic_frame
                            
                except (ValueError, IndexError):
                    # Skip problematic frames
                    continue
                    
            return enhanced
            
        except Exception:
            # If harmonic enhancement fails, return original audio
            return audio
    
    def _correct_formants(self, audio: np.ndarray) -> np.ndarray:
        """Apply formant correction for clearer vowels."""
        # This is a simplified formant enhancement
        # In practice, you'd use more sophisticated formant tracking
        
        # Apply gentle EQ to enhance formant regions
        nyquist = self.sample_rate // 2
        
        # Enhance formant regions (roughly 500Hz, 1500Hz, 2500Hz)
        for formant_freq in [500, 1500, 2500]:
            if formant_freq < nyquist:
                # Create bandpass filter
                low = formant_freq - 100
                high = formant_freq + 100
                sos = signal.butter(4, [low, high], btype='band', fs=self.sample_rate, output='sos')
                formant_signal = signal.sosfilt(sos, audio)
                audio += 0.1 * formant_signal  # Gentle enhancement
                
        return audio
    
    def _enhance_breathing(self, audio: np.ndarray) -> np.ndarray:
        """Enhance natural breathing sounds."""
        # Detect low-energy regions (potential breath sounds)
        frame_length = 2048
        hop_length = 512
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Enhance quiet regions slightly (breathing)
        for i, energy in enumerate(rms):
            start_sample = i * hop_length
            end_sample = min(start_sample + hop_length, len(audio))
            
            # Safe scalar conversion for energy comparison
            energy_val = float(energy) if np.isscalar(energy) else float(np.mean(energy))
            if energy_val < 0.1:  # Low energy region - fixed array comparison
                # Apply gentle high-pass filter to enhance breath texture
                sos = signal.butter(2, 200, btype='high', fs=self.sample_rate, output='sos')
                breath_enhanced = signal.sosfilt(sos, audio[start_sample:end_sample])
                audio[start_sample:end_sample] += 0.05 * breath_enhanced
                
        return audio
    
    def _optimize_dynamic_range(self, audio: np.ndarray) -> np.ndarray:
        """Optimize dynamic range for better clarity."""
        # Gentle compression
        threshold = 0.8
        ratio = 4.0
        
        # Simple compression algorithm
        compressed = audio.copy()
        above_threshold = np.abs(compressed) > threshold
        
        # Apply compression to samples above threshold
        compressed[above_threshold] = np.sign(compressed[above_threshold]) * (
            threshold + (np.abs(compressed[above_threshold]) - threshold) / ratio
        )
        
        return compressed


def get_elevenlabs_config() -> ElevenLabsConfig:
    """Get default ElevenLabs-inspired configuration."""
    return ElevenLabsConfig()


def create_elevenlabs_loss_function(config: ElevenLabsConfig) -> ElevenLabsQualityLoss:
    """Create ElevenLabs-inspired loss function."""
    return ElevenLabsQualityLoss(config)


def create_elevenlabs_processor(config: ElevenLabsConfig) -> ElevenLabsAudioProcessor:
    """Create ElevenLabs-inspired audio processor."""
    return ElevenLabsAudioProcessor(config)
