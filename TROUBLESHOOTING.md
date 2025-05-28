# TTS Audio Quality Troubleshooting Guide

This guide addresses common audio quality issues in TTS training and provides solutions.

## 🎯 Issues Addressed

### 1. Audio Cutting Off (Missing Words)

**Symptoms:**
- Generated audio ends abruptly
- Last words of sentences are missing
- Audio seems shorter than expected

**Root Causes & Solutions:**

#### Aggressive Silence Trimming
```python
# ❌ Problem: Too aggressive trimming
trim_db: int = 45  # Too high, cuts actual speech

# ✅ Solution: Gentler trimming  
trim_db: int = 30  # Lower threshold, preserves speech endings
```

#### Text Length Limitations
```python
# ❌ Problem: Text too long for model
max_text_length=200  # May truncate longer texts

# ✅ Solution: Increased limits
max_text_length=300  # Allows longer texts
```

#### Audio Length Constraints
```python
# ❌ Problem: Audio length too restrictive
max_wav_length=255995  # ~11.6 seconds

# ✅ Solution: Longer audio support
max_wav_length=440000  # ~20 seconds
```

### 2. Robotic/Interference Sounds

**Symptoms:**
- Metallic, artificial-sounding voice
- Digital artifacts in audio
- Unnatural prosody

**Root Causes & Solutions:**

#### Poor Spectral Resolution
```python
# ❌ Problem: Low resolution
fft_size: int = 1024
num_mels: int = 80

# ✅ Solution: Higher resolution
fft_size: int = 2048    # Better frequency resolution
num_mels: int = 100     # More mel channels
```

#### Insufficient Griffin-Lim Iterations
```python
# ❌ Problem: Too few iterations
griffin_lim_iters: int = 60

# ✅ Solution: More iterations
griffin_lim_iters: int = 100  # Better audio reconstruction
```

#### Poor Normalization
```python
# ❌ Problem: Inconsistent audio levels
do_sound_norm: bool = False
do_rms_norm: bool = False

# ✅ Solution: Better normalization
do_sound_norm: bool = True
do_rms_norm: bool = True
db_level: float = -25.0
```

### 3. Wrong Language Pronunciation

**Symptoms:**
- Model speaks words in wrong language (e.g., Chinese)
- Mixed language output
- Incorrect phonetic pronunciation

**Root Causes & Solutions:**

#### Dataset Language Mixing
```python
# ✅ Solution: Filter training data by language
def filter_by_language(samples, target_language):
    return [s for s in samples if s.get('language') == target_language]
```

#### Insufficient Language-Specific Data
```python
# ✅ Solution: Text length filtering for quality
def filter_samples_by_text_length(samples, min_words=5, max_words=50):
    filtered = []
    for sample in samples:
        word_count = len(sample['text'].split())
        if min_words <= word_count <= max_words:
            filtered.append(sample)
    return filtered
```

## 🔧 Improved Configuration Summary

### Audio Processing Improvements
```python
@dataclass
class ImprovedAudioConfig(BaseAudioConfig):
    # Better spectral resolution
    fft_size: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    
    # Gentle silence handling
    trim_db: int = 30  # Less aggressive
    
    # Better normalization
    do_sound_norm: bool = True
    do_rms_norm: bool = True
    db_level: float = -25.0
    
    # Improved reconstruction
    griffin_lim_iters: int = 100
    power: float = 2.0
    
    # Better mel-spectrogram
    num_mels: int = 100
    mel_fmin: float = 50.0
    mel_fmax: float = 8000.0
```

### Training Improvements
```python
# Longer audio support
max_conditioning_length=220000  # 10 seconds
max_wav_length=440000          # 20 seconds
max_text_length=300            # Longer texts

# Better optimization
optimizer_params={
    "betas": [0.9, 0.999],    # More stable
    "weight_decay": 5e-3      # Reduced regularization
}
lr=3e-06                      # Lower learning rate
lr_scheduler="CosineAnnealingLR"  # Smoother decay

# Data filtering
text_length_range=(5, 50)     # Filter extreme lengths
```

## 🎵 Quality Checklist

### Pre-Training
- [ ] Audio files are high quality (16+ kHz, mono)
- [ ] Text transcriptions are accurate
- [ ] No background noise or artifacts
- [ ] Consistent speaker voice
- [ ] Language consistency across dataset

### During Training
- [ ] Monitor loss curves for smooth convergence
- [ ] Listen to generated samples periodically
- [ ] Check for overfitting (eval vs train loss)
- [ ] Validate checkpoint quality regularly

### Post-Training
- [ ] Test with various text lengths
- [ ] Verify language consistency
- [ ] Check for robotic artifacts
- [ ] Validate word completion
- [ ] Test edge cases (long/short texts)

## 🛠️ Debugging Commands

### Check Audio Quality
```python
# Analyze audio file properties
import librosa
import numpy as np

def analyze_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))
    
    print(f"Duration: {duration:.2f}s")
    print(f"Sample Rate: {sr} Hz")
    print(f"RMS Level: {20*np.log10(rms):.1f} dB")
    print(f"Dynamic Range: {np.max(audio) - np.min(audio):.3f}")
```

### Validate Training Data
```python
# Check text/audio alignment
def validate_dataset(csv_path):
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Text length distribution
    text_lengths = df['text'].str.split().str.len()
    print(f"Text length stats: {text_lengths.describe()}")
    
    # Check for missing files
    missing = 0
    for audio_file in df['audio_file']:
        if not os.path.exists(audio_file):
            missing += 1
    
    print(f"Missing audio files: {missing}/{len(df)}")
```

## 📊 Performance Monitoring

### Key Metrics to Watch
1. **Training Loss**: Should decrease smoothly
2. **Evaluation Loss**: Should follow training loss
3. **Audio Quality**: Listen to samples regularly
4. **Memory Usage**: Monitor GPU/CPU usage
5. **Training Speed**: Steps per second

### Warning Signs
- Loss oscillating wildly → Reduce learning rate
- Eval loss increasing → Overfitting, reduce epochs
- Robotic output → Check audio config
- Cut-off words → Adjust silence trimming
- Wrong language → Check dataset filtering

## 🎯 Expected Results

With the improved configuration, you should see:
- ✅ Complete words/sentences (no cutting)
- ✅ More natural, less robotic sound
- ✅ Consistent target language pronunciation
- ✅ Better overall audio quality
- ✅ Smoother training convergence
