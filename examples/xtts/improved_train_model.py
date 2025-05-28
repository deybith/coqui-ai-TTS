"""
Improved training model wrapper with better audio quality settings.
"""

import os
import traceback

import torch

from xtts.improved_gpt_trainer import train_gpt_improved


def clear_gpu_cache():
    """Clear GPU cache to prevent memory issues."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_model_improved(
    language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length
):
    """
    Improved training function with better quality settings.
    
    Args:
        language: Target language for training
        train_csv: Path to training CSV file  
        eval_csv: Path to evaluation CSV file
        num_epochs: Number of training epochs
        batch_size: Training batch size
        grad_acumm: Gradient accumulation steps
        output_path: Output directory path
        max_audio_length: Maximum audio length in seconds
        
    Returns:
        Tuple of training results or error message
    """
    clear_gpu_cache()
    
    if not train_csv or not eval_csv:
        return (
            "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !",
            "",
            "",
            "",
            "",
        )
    
    try:
        # Convert seconds to waveform frames with improved length
        # Using 22050 Hz sample rate and allowing longer audio (20 seconds max instead of 11.6)
        max_audio_length_samples = int(max_audio_length * 22050)
        
        # Ensure minimum reasonable length
        if max_audio_length_samples < 22050:  # Less than 1 second
            max_audio_length_samples = 22050 * 10  # Default to 10 seconds
            
        # Cap maximum length to prevent memory issues
        if max_audio_length_samples > 22050 * 30:  # More than 30 seconds
            max_audio_length_samples = 22050 * 30  # Cap at 30 seconds
            
        print(f"Using max audio length: {max_audio_length_samples} samples ({max_audio_length_samples/22050:.1f} seconds)")
        
        config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt_improved(
            language,
            num_epochs,
            batch_size,
            grad_acumm,
            train_csv,
            eval_csv,
            output_path=output_path,
            max_audio_length=max_audio_length_samples,
        )
        
    except Exception as e:
        traceback.print_exc()
        error = traceback.format_exc()
        return (
            f"The training was interrupted due to an error! Please check the console for the full error message!\nError summary: {error}",
            "",
            "",
            "",
            "",
        )

    # copy original files to avoid parameter change issues
    os.system(f"cp {config_path} {exp_path}")
    os.system(f"cp {vocab_file} {exp_path}")

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    print("Improved model training done!")
    clear_gpu_cache()
    
    return "Improved model training completed successfully!", config_path, vocab_file, ft_xtts_checkpoint, speaker_wav


def validate_training_data(train_csv, eval_csv):
    """
    Validate training data for common issues that cause audio problems.
    
    Args:
        train_csv: Path to training CSV file
        eval_csv: Path to evaluation CSV file
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    try:
        import pandas as pd
        
        # Load training data
        train_df = pd.read_csv(train_csv)
        eval_df = pd.read_csv(eval_csv)
        
        # Check for minimum data requirements
        if len(train_df) < 50:
            warnings.append(f"Warning: Training data has only {len(train_df)} samples. Recommend at least 100 samples for good quality.")
            
        if len(eval_df) < 10:
            warnings.append(f"Warning: Evaluation data has only {len(eval_df)} samples. Recommend at least 20 samples.")
        
        # Check text length distribution
        if 'text' in train_df.columns:
            text_lengths = train_df['text'].str.split().str.len()
            very_short = (text_lengths < 3).sum()
            very_long = (text_lengths > 100).sum()
            
            if very_short > len(train_df) * 0.1:
                warnings.append(f"Warning: {very_short} training samples have very short text (<3 words). This may cause cutting issues.")
                
            if very_long > len(train_df) * 0.1:
                warnings.append(f"Warning: {very_long} training samples have very long text (>100 words). This may cause memory issues.")
        
        # Check for language consistency (if language info available)
        if 'language' in train_df.columns:
            languages = train_df['language'].unique()
            if len(languages) > 1:
                warnings.append(f"Warning: Multiple languages detected in training data: {languages}. This may cause language mixing issues.")
        
        # Check audio file existence (if paths are provided)
        if 'audio_file' in train_df.columns:
            missing_files = 0
            for audio_file in train_df['audio_file'].head(10):  # Check first 10 files
                if not os.path.exists(audio_file):
                    missing_files += 1
            
            if missing_files > 0:
                warnings.append(f"Warning: {missing_files}/10 sampled audio files are missing. Check audio file paths.")
    
    except Exception as e:
        warnings.append(f"Error validating training data: {e}")
    
    return warnings


def print_training_tips():
    """Print helpful tips for better training results."""
    tips = """
    🎯 TRAINING TIPS FOR BETTER AUDIO QUALITY:
    
    📝 Data Preparation:
    • Use high-quality audio files (16-24 kHz, mono, minimal background noise)
    • Ensure text transcriptions are accurate and match the audio exactly
    • Keep text lengths reasonable (5-50 words per sample)
    • Remove samples with excessive silence at start/end
    
    🔧 Training Settings:
    • Start with fewer epochs (5-10) and gradually increase if needed
    • Use smaller batch sizes if experiencing memory issues
    • Monitor loss curves - training should converge smoothly
    
    🎵 Audio Quality:
    • The improved configuration reduces robotic artifacts
    • Less aggressive silence trimming prevents word cutting
    • Better normalization ensures consistent volume
    
    🌍 Language Issues:
    • Ensure all training data is in the target language
    • Check that tokenizer supports your target language properly
    • Consider language-specific fine-tuning if mixing occurs
    
    📊 Monitoring:
    • Watch for overfitting (eval loss increasing while train loss decreases)
    • Listen to generated samples during training to catch issues early
    • Save checkpoints frequently to avoid losing progress
    """
    print(tips)
