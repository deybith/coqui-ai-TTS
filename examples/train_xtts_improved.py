"""
Improved XTTS training script with better audio quality settings.
This script addresses common issues like audio cutting, robotic sounds, and language mixing.
"""

from xtts.improved_train_model import train_model_improved, validate_training_data, print_training_tips

# Training configuration
lang = 'es'  # Change this to your target language
train_csv = '/home/ubuntu/projects/coqui-ai-Trainer/data/test/dataset/metadata_train.csv'
eval_csv = '/home/ubuntu/projects/coqui-ai-Trainer/data/test/dataset/metadata_eval.csv'
num_epochs = 15  # Increased from 10 for better convergence
batch_size = 8   # Reduced from 16 for better quality and memory usage
grad_acumm = 2   # Increased gradient accumulation to maintain effective batch size
out_path = '/home/ubuntu/projects/coqui-ai-Trainer/data/test'
max_audio_length = 20  # Increased from 30 seconds to allow longer utterances


def main():
    """Main training function with improved settings."""
    
    print("🎤 Starting Improved XTTS Training")
    print("=" * 50)
    
    # Print helpful tips
    print_training_tips()
    
    # Validate training data first
    print("\n🔍 Validating training data...")
    warnings = validate_training_data(train_csv, eval_csv)
    
    if warnings:
        print("⚠️  Data validation warnings:")
        for warning in warnings:
            print(f"   • {warning}")
        print()
    else:
        print("✅ Training data validation passed!")
    
    # Start training with improved settings
    print(f"\n🚀 Starting training with improved configuration:")
    print(f"   • Language: {lang}")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Gradient accumulation: {grad_acumm}")
    print(f"   • Max audio length: {max_audio_length}s")
    print(f"   • Output path: {out_path}")
    print()
    
    # Run improved training
    result = train_model_improved(
        lang,
        train_csv,
        eval_csv,
        num_epochs,
        batch_size,
        grad_acumm,
        out_path,
        max_audio_length
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("🎯 TRAINING RESULTS:")
    print("=" * 50)
    
    if isinstance(result, tuple) and len(result) == 5:
        status, config_path, vocab_file, checkpoint_path, speaker_wav = result
        
        if "successfully" in status.lower():
            print("✅ Training completed successfully!")
            print(f"📁 Config file: {config_path}")
            print(f"📝 Vocab file: {vocab_file}")
            print(f"🎯 Best checkpoint: {checkpoint_path}")
            print(f"🎵 Speaker reference: {speaker_wav}")
            
            print("\n🎉 IMPROVEMENTS APPLIED:")
            print("   • Reduced silence trimming (less word cutting)")
            print("   • Improved audio normalization (less robotic sound)")
            print("   • Better text length filtering (prevents truncation)")
            print("   • Enhanced training stability (smoother convergence)")
            print("   • More frequent checkpointing (better monitoring)")
            
        else:
            print("❌ Training failed:")
            print(f"   Error: {status}")
    else:
        print("❌ Unexpected training result format")
        print(f"   Result: {result}")


if __name__ == "__main__":
    main()
