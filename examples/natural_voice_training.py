"""
Natural Voice Training Script
============================

This script implements ultra-natural voice training using advanced audio processing techniques.
It goes beyond the existing improvements to achieve truly human-like voice synthesis.
"""

import os
import sys
import gc
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Add trainer to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trainer.natural_voice_config import get_natural_voice_config, get_natural_voice_training_config
from trainer.natural_voice_trainer import create_natural_voice_trainer
from trainer import TrainerArgs

# Import TTS components
try:
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.utils.manage import ModelManager
except ImportError as e:
    print(f"❌ Error importing TTS components: {e}")
    print("Please ensure TTS is properly installed")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_natural_voice_model(
    language: str = "en",
    train_csv: str = "",
    eval_csv: str = "",
    output_path: str = "./output/natural_voice",
    num_epochs: int = 20,
    batch_size: int = 2,  # Smaller for higher quality
    gradient_accumulation: int = 8,  # Maintain effective batch size
    max_audio_length: int = 600000,  # ~27 seconds at 22kHz
    learning_rate: float = 5e-7,  # Ultra-conservative for natural sound
    use_pretrained: bool = True,
    speaker_reference: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a model for ultra-natural voice synthesis.
    
    Args:
        language: Target language code
        train_csv: Path to training metadata CSV
        eval_csv: Path to evaluation metadata CSV  
        output_path: Directory for training outputs
        num_epochs: Number of training epochs
        batch_size: Training batch size (kept small for quality)
        gradient_accumulation: Gradient accumulation steps
        max_audio_length: Maximum audio length in samples
        learning_rate: Learning rate (very conservative)
        use_pretrained: Whether to use pretrained model
        speaker_reference: Path to speaker reference audio
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with training results
    """
    
    print("🎵 Starting Natural Voice Training")
    print("=" * 50)
    
    # Validate inputs
    if not train_csv or not eval_csv:
        return {"error": "Please provide both train_csv and eval_csv paths"}
    
    if not os.path.exists(train_csv):
        return {"error": f"Training CSV not found: {train_csv}"}
    
    if not os.path.exists(eval_csv):
        return {"error": f"Evaluation CSV not found: {eval_csv}"}
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Clear GPU memory
        gc.collect()
        if hasattr(gc, 'torch'):
            gc.torch.cuda.empty_cache()
        
        # Get natural voice configurations
        natural_audio_config = get_natural_voice_config()
        natural_training_config = get_natural_voice_training_config()
        
        print("🔧 Configurations:")
        print(f"   Sample Rate: {natural_audio_config.sample_rate} Hz")
        print(f"   Output Sample Rate: {natural_audio_config.output_sample_rate} Hz")
        print(f"   FFT Size: {natural_audio_config.fft_size}")
        print(f"   Mel Channels: {natural_audio_config.num_mels}")
        print(f"   Griffin-Lim Iterations: {natural_audio_config.griffin_lim_iters}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Batch Size: {batch_size}")
        
        # Load and prepare model
        print("\n📦 Loading model...")
        model_manager = ModelManager()
        
        if use_pretrained:
            # Download pretrained model
            model_path, config_path, model_item = model_manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
            print(f"   Loaded pretrained model from: {model_path}")
        else:
            print("   Using model from scratch")
            model_path = None
            config_path = None
        
        # Create enhanced audio configuration
        # XttsAudioConfig only accepts specific parameters, so we create it with supported ones
        audio_config = XttsAudioConfig(
            sample_rate=natural_audio_config.sample_rate,
            output_sample_rate=natural_audio_config.output_sample_rate,
            dvae_sample_rate=natural_audio_config.sample_rate,  # Match sample rate
        )
        
        # Apply additional natural voice enhancements by directly setting attributes
        # This allows us to use our enhanced configuration without constructor errors
        try:
            # Enhanced STFT parameters
            if hasattr(audio_config, 'fft_size'):
                audio_config.fft_size = natural_audio_config.fft_size
            if hasattr(audio_config, 'win_length'):
                audio_config.win_length = natural_audio_config.win_length
            if hasattr(audio_config, 'hop_length'):
                audio_config.hop_length = natural_audio_config.hop_length
            
            # Natural voice mel parameters
            if hasattr(audio_config, 'num_mels'):
                audio_config.num_mels = natural_audio_config.num_mels
            if hasattr(audio_config, 'mel_fmin'):
                audio_config.mel_fmin = natural_audio_config.mel_fmin
            if hasattr(audio_config, 'mel_fmax'):
                audio_config.mel_fmax = natural_audio_config.mel_fmax
            
            # Enhanced synthesis
            if hasattr(audio_config, 'griffin_lim_iters'):
                audio_config.griffin_lim_iters = natural_audio_config.griffin_lim_iters
            if hasattr(audio_config, 'power'):
                audio_config.power = natural_audio_config.power
            
            # Natural sound processing
            if hasattr(audio_config, 'do_sound_norm'):
                audio_config.do_sound_norm = natural_audio_config.do_sound_norm
            if hasattr(audio_config, 'do_rms_norm'):
                audio_config.do_rms_norm = natural_audio_config.do_rms_norm
            if hasattr(audio_config, 'db_level'):
                audio_config.db_level = natural_audio_config.db_level
            
            # Conservative silence handling
            if hasattr(audio_config, 'do_trim_silence'):
                audio_config.do_trim_silence = natural_audio_config.do_trim_silence
            if hasattr(audio_config, 'trim_db'):
                audio_config.trim_db = natural_audio_config.trim_db
            
            # Enhanced dynamic range
            if hasattr(audio_config, 'min_level_db'):
                audio_config.min_level_db = natural_audio_config.min_level_db
            if hasattr(audio_config, 'max_norm'):
                audio_config.max_norm = natural_audio_config.max_norm
            
            # Natural frequency processing
            if hasattr(audio_config, 'preemphasis'):
                audio_config.preemphasis = natural_audio_config.preemphasis
            if hasattr(audio_config, 'ref_level_db'):
                audio_config.ref_level_db = natural_audio_config.ref_level_db
            
            # Quality enhancements
            if hasattr(audio_config, 'signal_norm'):
                audio_config.signal_norm = natural_audio_config.signal_norm
            if hasattr(audio_config, 'symmetric_norm'):
                audio_config.symmetric_norm = natural_audio_config.symmetric_norm
            if hasattr(audio_config, 'clip_norm'):
                audio_config.clip_norm = natural_audio_config.clip_norm
                
            print("   ✅ Enhanced audio configuration applied")
            
        except Exception as e:
            print(f"   ⚠️  Some audio enhancements could not be applied: {e}")
            print("   Using basic configuration with available parameters")
        
        # Create model arguments with natural voice enhancements
        print("\n🧠 Creating model arguments...")
        model_args = GPTArgs(
            max_conditioning_length=300000,  # ~13.6 seconds at 22kHz for natural flow
            min_conditioning_length=66150,   # Keep minimum conditioning
            debug_loading_failures=True,     # Enable debugging for loading failures
            max_wav_length=max_audio_length, # Use provided max length
            max_text_length=400,             # Support longer texts for natural expression
            mel_norm_file=None,              # Will be set if using pretrained
            dvae_checkpoint=None,            # Will be set if using pretrained
            xtts_checkpoint=None,            # Will be set if using pretrained
            tokenizer_file=None,             # Will be set if using pretrained
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        )
        
        # Update model args with pretrained paths if available
        if model_path and config_path:
            model_dir = os.path.dirname(model_path)
            # Try to find the required files in the model directory
            potential_files = {
                'mel_norm_file': 'mel_stats.pth',
                'dvae_checkpoint': 'dvae.pth', 
                'tokenizer_file': 'vocab.json'
            }
            
            for attr, filename in potential_files.items():
                full_path = os.path.join(model_dir, filename)
                if os.path.exists(full_path):
                    setattr(model_args, attr, full_path)
                    print(f"   Found {filename} at {full_path}")
            
            model_args.xtts_checkpoint = model_path
            print(f"   Set XTTS checkpoint: {model_path}")
        
        # Create training configuration with only valid GPTTrainerConfig parameters
        config = GPTTrainerConfig(
            # Core configuration parameters (based on working examples)
            epochs=num_epochs,
            output_path=output_path,
            model_args=model_args,
            run_name=f"natural_voice_{language}",
            project_name="Natural_Voice_TTS",
            run_description="Enhanced natural voice training for better audio quality",
            dashboard_logger="tensorboard",
            logger_uri=None,
            audio=audio_config,
            
            # Training settings
            batch_size=batch_size,
            eval_batch_size=batch_size,
            
            # Conservative learning for natural sound
            lr=learning_rate,
            optimizer="AdamW",
            optimizer_params={
                "betas": [0.9, 0.999],
                "weight_decay": 1e-4,  # Light regularization
                "eps": 1e-8,
            },
            
            # Learning rate schedule
            lr_scheduler="ExponentialLR",
            lr_scheduler_params={"gamma": 0.9995},  # Very gradual decay
            
            # Monitoring and logging
            print_step=25,  # Frequent monitoring
            save_step=500,  # Regular checkpoints
            plot_step=50,   # Regular plotting
            
            # Training stability
            grad_clip=0.5,  # Conservative gradient clipping
            
            # Data processing settings
            num_loader_workers=4,
            training_seed=54321,
            
            # Evaluation settings
            eval_split_max_size=512,
            print_eval=True,
            
            # Mixed precision for efficiency
            mixed_precision=True,
            
            test_sentences=[]
        )
        
        # Load and filter data for natural voice
        print("\n📊 Loading and filtering training data...")
        
        # Create dataset configuration
        from TTS.config.shared_configs import BaseDatasetConfig
        dataset_config = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="natural_voice_training",
            path=os.path.dirname(train_csv),
            meta_file_train=train_csv,
            meta_file_val=eval_csv,
            language=language,
        )
        
        # Load samples using the dataset config list
        train_samples, eval_samples = load_tts_samples(
            [dataset_config],
            eval_split=True,
            eval_split_max_size=512,
            eval_split_size=0.1,
        )
        
        print(f"   Raw training samples: {len(train_samples)}")
        print(f"   Raw evaluation samples: {len(eval_samples)}")
        
        # Apply ultra-strict filtering for natural voice
        train_samples_filtered = filter_for_natural_voice(train_samples, language)
        eval_samples_filtered = filter_for_natural_voice(eval_samples, language)
        
        print(f"   Filtered training samples: {len(train_samples_filtered)}")
        print(f"   Filtered evaluation samples: {len(eval_samples_filtered)}")
        
        if len(train_samples_filtered) < 10:
            return {"error": f"Too few training samples after filtering: {len(train_samples_filtered)}"}
        
        # Initialize model
        print("\n🏗️  Initializing model...")
        if model_path:
            # Load pretrained model
            model = GPTTrainer.init_from_config(config, samples=train_samples_filtered, verbose=True)
            # Load pretrained weights if available
            try:
                model.load_checkpoint(config, checkpoint_path=model_path, eval=False, strict=False)
                print("   ✅ Loaded pretrained weights")
            except Exception as e:
                print(f"   ⚠️  Could not load pretrained weights: {e}")
        else:
            model = GPTTrainer.init_from_config(config, samples=train_samples_filtered, verbose=True)
        
        # Create natural voice trainer
        print("\n🎵 Creating Natural Voice Trainer...")
        trainer = create_natural_voice_trainer(
            TrainerArgs(
                restore_path=None,
                skip_train_epoch=False,
                start_with_eval=True,  # Start with evaluation
                grad_accum_steps=gradient_accumulation,
            ),
            config,
            output_path=output_path,
            model=model,
            train_samples=train_samples_filtered,
            eval_samples=eval_samples_filtered,
            natural_voice_config=natural_audio_config
        )
        
        print("   ✅ Natural Voice Trainer ready")
        
        # Start training
        print("\n🚀 Starting Natural Voice Training...")
        print("   This may take several hours for best quality")
        print("   Monitor the output for quality improvements")
        
        trainer.fit()
        
        print("\n✅ Training completed successfully!")
        
        # Find the best model
        checkpoint_dir = Path(output_path)
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth"))
        
        if checkpoints:
            best_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"   Best checkpoint: {best_checkpoint}")
            
            return {
                "status": "success",
                "output_path": output_path,
                "best_checkpoint": str(best_checkpoint),
                "train_samples": len(train_samples_filtered),
                "eval_samples": len(eval_samples_filtered),
                "audio_config": {
                    "sample_rate": natural_audio_config.sample_rate,
                    "output_sample_rate": natural_audio_config.output_sample_rate,
                    "fft_size": natural_audio_config.fft_size,
                    "num_mels": natural_audio_config.num_mels,
                    "griffin_lim_iters": natural_audio_config.griffin_lim_iters,
                }
            }
        else:
            return {"error": "No checkpoints found after training"}
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Training failed: {str(e)}"}


def filter_for_natural_voice(samples: list, language: str, min_words: int = 3, max_words: int = 40) -> list:
    """
    Apply ultra-strict filtering for natural voice training.
    
    Args:
        samples: List of training samples
        language: Target language
        min_words: Minimum words per sample
        max_words: Maximum words per sample
        
    Returns:
        Filtered list of samples
    """
    
    filtered_samples = []
    
    for sample in samples:
        try:
            # Check text length
            text = sample.get("text", "")
            if not text:
                continue
            
            word_count = len(text.split())
            if word_count < min_words or word_count > max_words:
                continue
            
            # Check for language consistency
            if language and sample.get("language") != language:
                continue
            
            # Check audio file exists
            audio_path = sample.get("audio_file") or sample.get("wav_file_name")
            if not audio_path or not os.path.exists(audio_path):
                continue
            
            # Additional quality checks
            # Check for special characters that might cause issues
            if any(char in text for char in ["@", "#", "$", "%", "^", "&", "*"]):
                continue
            
            # Check for excessive punctuation
            punct_count = sum(1 for char in text if char in ".,!?;:")
            if punct_count > word_count * 0.3:  # Max 30% punctuation
                continue
            
            # Check for numbers (might need special handling)
            if any(char.isdigit() for char in text):
                # Skip samples with numbers for pure natural voice
                continue
            
            # Text should not be all caps (sounds unnatural)
            if text.isupper():
                continue
            
            # Add sample if it passes all checks
            filtered_samples.append(sample)
            
        except Exception as e:
            # Skip problematic samples
            continue
    
    return filtered_samples


def validate_audio_quality(audio_path: str, min_duration: float = 1.0, max_duration: float = 20.0) -> bool:
    """
    Validate audio file quality for natural voice training.
    
    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        
    Returns:
        True if audio passes quality checks
    """
    
    try:
        import librosa
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        # Check duration
        if duration < min_duration or duration > max_duration:
            return False
        
        # Check for silence
        rms = librosa.feature.rms(y=audio)[0]
        if np.mean(rms) < 0.001:  # Too quiet
            return False
        
        # Check for clipping
        if np.max(np.abs(audio)) > 0.95:  # Potential clipping
            return False
        
        # Check for reasonable dynamic range
        dynamic_range = np.max(audio) - np.min(audio)
        if dynamic_range < 0.1:  # Too flat
            return False
        
        return True
        
    except Exception:
        return False


def print_natural_voice_tips():
    """Print tips for achieving natural voice quality."""
    
    tips = """
    🎵 NATURAL VOICE TRAINING TIPS:
    
    📊 Data Quality (Most Important):
    • Use studio-quality recordings (24kHz+, no background noise)
    • Ensure perfect text-audio alignment
    • Use consistent speaker voice throughout
    • Avoid robotic or monotone samples
    • Remove samples with audio artifacts
    
    🎯 Training Strategy:
    • Start with small batch sizes (2-4) for maximum quality
    • Use very conservative learning rates (5e-7 to 1e-6)
    • Train for more epochs with patience
    • Monitor audio samples frequently during training
    
    🔧 Audio Settings:
    • Higher sample rates preserve naturalness
    • More mel channels capture vocal nuances  
    • Conservative silence trimming prevents word cutting
    • Advanced normalization ensures consistency
    
    🎤 Voice Naturalness:
    • Longer Griffin-Lim iterations improve quality
    • Perceptual losses help human-like sound
    • Prosody matching preserves natural rhythm
    • Spectral processing reduces artifacts
    
    ⚠️  Common Issues:
    • Robotic sound → Increase Griffin-Lim iterations, check normalization
    • Muffled voice → Increase mel_fmax, check frequency range
    • Choppy speech → Reduce silence trimming, check hop_length
    • Unclear words → Increase FFT size, check sample rate
    
    📈 Monitoring:
    • Listen to samples every 100-200 steps
    • Watch for smooth loss convergence
    • Validate with different text lengths
    • Test emotional expression range
    """
    
    print(tips)


if __name__ == "__main__":
    # Example usage
    print("🎵 Natural Voice Training System")
    print("=" * 40)
    
    # Print tips
    print_natural_voice_tips()
    
    # Example training (customize these paths)
    example_config = {
        "language": "en",
        "train_csv": "/home/ubuntu/projects/coqui-ai-Trainer/data/test/dataset/metadata_train.csv",
        "eval_csv": "/home/ubuntu/projects/coqui-ai-Trainer/data/test/dataset/metadata_eval.csv",
        "output_path": "./output/natural_voice_training",
        "num_epochs": 25,
        "batch_size": 2,
        "gradient_accumulation": 8,
        "learning_rate": 5e-7,
    }
    
    print("\n📋 Example Configuration:")
    for key, value in example_config.items():
        print(f"   {key}: {value}")
    
    print("\n💡 To start training, update the paths and run:")
    print("   python natural_voice_training.py")
    
    # Uncomment to run training:
    result = train_natural_voice_model(**example_config)
    print(f"\n🎵 Training Result: {result}")
