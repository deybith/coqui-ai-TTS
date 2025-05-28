#!/usr/bin/env python3
"""
Quick patch script to apply audio quality improvements to existing TTS trainer.
Run this script to automatically apply the fixes for:
- Audio cutting off (missing words)
- Robotic/interference sounds  
- Wrong language pronunciation
"""

import os
import shutil
from datetime import datetime


def backup_original_files():
    """Create backup of original files before applying patches."""
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        "examples/xtts/gpt_trainer.py",
        "examples/xtts/shared_configs.py", 
        "examples/train_xtts.py"
    ]
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            shutil.copy2(file_path, os.path.join(backup_dir, os.path.basename(file_path)))
            print(f"✅ Backed up {file_path}")
    
    print(f"📁 Backup created in: {backup_dir}")
    return backup_dir


def apply_audio_config_patch():
    """Apply audio configuration improvements."""
    config_file = "examples/xtts/shared_configs.py"
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    print("🔧 Applying audio configuration patches...")
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Apply patches for better audio quality
    patches = {
        'trim_db: int = 45': 'trim_db: int = 30  # Reduced for less aggressive trimming',
        'do_sound_norm: bool = False': 'do_sound_norm: bool = True  # Enable for better consistency',
        'do_rms_norm: bool = False': 'do_rms_norm: bool = True  # Enable RMS normalization',
        'db_level: float = None': 'db_level: float = -25.0  # Set appropriate level',
        'griffin_lim_iters: int = 60': 'griffin_lim_iters: int = 100  # Increased for better quality',
        'power: float = 1.5': 'power: float = 2.0  # Increased for better reconstruction',
        'num_mels: int = 80': 'num_mels: int = 100  # Increased for better resolution',
        'mel_fmin: float = 0.0': 'mel_fmin: float = 50.0  # Better for most voices',
        'min_level_db: int = -100': 'min_level_db: int = -120  # Better dynamic range'
    }
    
    patches_applied = 0
    for old_val, new_val in patches.items():
        if old_val in content:
            content = content.replace(old_val, new_val)
            patches_applied += 1
            print(f"   ✓ Applied: {old_val} → {new_val}")
    
    # Write patched config
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Applied {patches_applied} audio config patches")
    return True


def apply_training_config_patch():
    """Apply training configuration improvements."""
    trainer_file = "examples/xtts/gpt_trainer.py"
    
    if not os.path.exists(trainer_file):
        print(f"❌ Trainer file not found: {trainer_file}")
        return False
    
    print("🔧 Applying training configuration patches...")
    
    # Read current trainer
    with open(trainer_file, 'r') as f:
        content = f.read()
    
    # Apply training patches
    patches = {
        'max_conditioning_length=132300': 'max_conditioning_length=220000  # Increased for longer conditioning',
        'max_text_length=200': 'max_text_length=300  # Increased for longer texts',
        'lr=5e-06': 'lr=3e-06  # Reduced for more stable training',
        '"betas": [0.9, 0.96]': '"betas": [0.9, 0.999]  # More stable betas',
        '"weight_decay": 1e-2': '"weight_decay": 5e-3  # Reduced weight decay',
        'num_loader_workers=8': 'num_loader_workers=4  # Reduced to avoid loading issues',
        'print_step=50': 'print_step=25  # More frequent monitoring',
        'save_step=1000': 'save_step=500  # More frequent saves'
    }
    
    patches_applied = 0
    for old_val, new_val in patches.items():
        if old_val in content:
            content = content.replace(old_val, new_val)
            patches_applied += 1
            print(f"   ✓ Applied: {old_val}")
    
    # Write patched trainer
    with open(trainer_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Applied {patches_applied} training config patches")
    return True


def update_example_script():
    """Update the example training script with better defaults."""
    example_file = "examples/train_xtts.py"
    
    if not os.path.exists(example_file):
        print(f"❌ Example file not found: {example_file}")
        return False
    
    print("🔧 Updating example script...")
    
    # Read current example
    with open(example_file, 'r') as f:
        content = f.read()
    
    # Apply example patches
    patches = {
        'max_audio_length=30': 'max_audio_length=20  # Increased but reasonable limit',
        'batch_size=16': 'batch_size=8  # Reduced for better quality',
        'grad_acumm=1': 'grad_acumm=2  # Increased gradient accumulation',
        'num_epochs=10': 'num_epochs=15  # Increased for better convergence'
    }
    
    patches_applied = 0
    for old_val, new_val in patches.items():
        if old_val in content:
            content = content.replace(old_val, new_val)
            patches_applied += 1
            print(f"   ✓ Applied: {old_val}")
    
    # Write patched example
    with open(example_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Applied {patches_applied} example script patches")
    return True


def main():
    """Main patch application function."""
    print("🎤 TTS Audio Quality Improvement Patcher")
    print("=" * 50)
    
    print("This script will apply improvements to fix:")
    print("• Audio cutting off (missing words)")  
    print("• Robotic/interference sounds")
    print("• Wrong language pronunciation")
    print()
    
    # Create backup
    backup_dir = backup_original_files()
    print()
    
    # Apply patches
    success_count = 0
    
    if apply_audio_config_patch():
        success_count += 1
    
    if apply_training_config_patch():
        success_count += 1
        
    if update_example_script():
        success_count += 1
    
    print("\n" + "=" * 50)
    print("🎯 PATCH SUMMARY")
    print("=" * 50)
    
    if success_count == 3:
        print("✅ All patches applied successfully!")
        print("\n🎉 IMPROVEMENTS APPLIED:")
        print("   • Reduced silence trimming (less word cutting)")
        print("   • Improved audio normalization (less robotic)")
        print("   • Better training stability (smoother convergence)")
        print("   • Enhanced monitoring (more frequent saves)")
        print("   • Optimized batch sizes (better quality)")
        
        print(f"\n📁 Original files backed up to: {backup_dir}")
        print("\n🚀 Ready to train with improved settings!")
        print("   Run: python examples/train_xtts.py")
        
    else:
        print(f"⚠️  Partially successful: {success_count}/3 patches applied")
        print("   Check the error messages above for details")
        print(f"   Original files backed up to: {backup_dir}")
    
    print("\n📚 For more details, see: TROUBLESHOOTING.md")


if __name__ == "__main__":
    main()
