# XTTS Ultra Enhanced Training - Usage Examples (REAL Improvements)

The `train_xtts_ultra_enhanced.py` script now uses **REAL audio quality improvements** and supports command line arguments for flexible configuration.

## ✅ What's Fixed

The script now uses **proven audio quality improvements** instead of fictional ones:
- **Better STFT analysis** (2048 FFT vs 1024) - Better frequency resolution
- **Gentler silence trimming** (30dB vs 45dB) - Prevents word cutting
- **Proper audio normalization** (sound + RMS) - Removes robotic sounds
- **Better reconstruction** (100 Griffin-Lim iterations vs 60) - Higher quality
- **Higher resolution** (100 mel bins vs 80) - More audio detail
- **Optimized frequency range** (50-8000Hz vs 0-8000Hz) - Better for voices
- **Text quality validation** - Prevents language mixing issues
- **Stable training parameters** - Proven optimizer settings

## Required Arguments

- `--train_csv`: Path to the training CSV metadata file
- `--eval_csv`: Path to the evaluation CSV metadata file  
- `--output_path`: Path to the output directory for training results

## Optional Arguments (with defaults)

- `--language`: Language code for training (default: "es")
- `--num_epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 6)
- `--grad_acumm`: Gradient accumulation steps (default: 3)
- `--max_audio_length`: Maximum audio length in samples (default: 480000)

## Usage Examples

### Basic Usage (using all defaults)
```bash
python train_xtts_ultra_enhanced.py \
    --train_csv "/path/to/dataset/metadata_train.csv" \
    --eval_csv "/path/to/dataset/metadata_eval.csv" \
    --output_path "/path/to/training/output"
```

### Custom Configuration
```bash
python train_xtts_ultra_enhanced.py \
    --train_csv "/home/ubuntu/projects/create-av-content/models/dieck/dataset/metadata_train.csv" \
    --eval_csv "/home/ubuntu/projects/create-av-content/models/dieck/dataset/metadata_eval.csv" \
    --output_path "/home/ubuntu/projects/create-av-content/models/dieck/run/training" \
    --language "en" \
    --num_epochs 30 \
    --batch_size 8 \
    --grad_acumm 2 \
    --max_audio_length 600000
```

### Quick Training (fewer epochs, smaller batch)
```bash
python train_xtts_ultra_enhanced.py \
    --train_csv "/path/to/dataset/metadata_train.csv" \
    --eval_csv "/path/to/dataset/metadata_eval.csv" \
    --output_path "/path/to/training/output" \
    --num_epochs 5 \
    --batch_size 2
```

### High-End Training (more epochs, larger batches)
```bash
python train_xtts_ultra_enhanced.py \
    --train_csv "/path/to/dataset/metadata_train.csv" \
    --eval_csv "/path/to/dataset/metadata_eval.csv" \
    --output_path "/path/to/training/output" \
    --num_epochs 50 \
    --batch_size 12 \
    --grad_acumm 4 \
    --max_audio_length 720000
```

## Expected Quality Improvements

With the **REAL improvements** in this script, you should notice:

### 🎤 **Audio Quality Improvements**
- **Clearer speech** - Better frequency analysis captures more detail
- **Complete words** - Gentler silence trimming prevents word cutting
- **Natural sound** - Proper normalization removes robotic artifacts
- **Smoother audio** - Better reconstruction reduces glitches and artifacts
- **Higher fidelity** - More mel bins capture subtle audio nuances
- **Cleaner output** - Optimized frequency range removes unwanted noise

### 📝 **Training Improvements**
- **No language mixing** - Text validation prevents pronunciation issues
- **Better convergence** - Stable optimizer settings improve training
- **Consistent quality** - RMS normalization ensures uniform audio levels
- **Fewer failures** - Quality filtering reduces problematic samples

### ⚙️ **Technical Improvements**
- **2x better frequency resolution** (2048 vs 1024 FFT)
- **25% more mel bins** (100 vs 80) for higher resolution
- **67% more reconstruction iterations** (100 vs 60 Griffin-Lim)
- **Better dynamic range** (-120dB vs -100dB)

## Troubleshooting

If you don't notice improvements, check:

1. **Verify real improvements are loaded**:
   ```bash
   python -c "
   import sys; sys.path.insert(0, 'examples')
   from train_xtts_ultra_enhanced import get_improved_audio_config
   config = get_improved_audio_config()
   print('FFT Size:', config['fft_size'])
   print('Trim DB:', config['trim_db'])
   print('Mel Bins:', config['num_mels'])
   "
   ```

2. **Compare with baseline** - Train a model with the original settings and compare

3. **Check your data quality** - The text validation will filter out poor samples

4. **Monitor training logs** - Look for stable loss curves and quality metrics

## Help
To see all available options:
```bash
python train_xtts_ultra_enhanced.py --help
```

## Migration from Previous Version
If you were using the script with hardcoded parameters, you can now achieve the same behavior by running:
```bash
python train_xtts_ultra_enhanced.py \
    --train_csv "/home/ubuntu/projects/create-av-content/models/dieck/dataset/metadata_train.csv" \
    --eval_csv "/home/ubuntu/projects/create-av-content/models/dieck/dataset/metadata_eval.csv" \
    --output_path "/home/ubuntu/projects/create-av-content/models/dieck/run/training" \
    --language "es" \
    --num_epochs 20 \
    --batch_size 6 \
    --grad_acumm 3 \
    --max_audio_length 480000
```

## Additional Files

- `REAL_vs_FAKE_IMPROVEMENTS.md` - Detailed comparison of fake vs real improvements
- `test_real_improvements.py` - Test script to verify improvements are working
