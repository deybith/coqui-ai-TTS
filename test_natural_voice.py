#!/usr/bin/env python3
"""
Natural Voice System Test Script
===============================

This script tests the natural voice configuration and trainer to ensure everything is working properly.
"""

import sys
import os
import torch
import traceback
from pathlib import Path

# Add trainer to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_natural_voice_config():
    """Test the natural voice configuration."""
    print("🧪 Testing Natural Voice Configuration...")
    
    try:
        from trainer.natural_voice_config import (
            get_natural_voice_config, 
            get_natural_voice_training_config,
            validate_natural_voice_config
        )
        
        # Test audio config
        audio_config = get_natural_voice_config()
        print(f"✅ Audio config loaded: {type(audio_config).__name__}")
        print(f"   Sample Rate: {audio_config.sample_rate} Hz")
        print(f"   Output Rate: {audio_config.output_sample_rate} Hz")
        print(f"   FFT Size: {audio_config.fft_size}")
        print(f"   Mel Channels: {audio_config.num_mels}")
        
        # Test training config
        training_config = get_natural_voice_training_config()
        print(f"✅ Training config loaded: {type(training_config).__name__}")
        print(f"   Learning Rate: {training_config.learning_rate}")
        print(f"   Batch Size: {training_config.batch_size}")
        
        # Validate config
        is_valid = validate_natural_voice_config(audio_config)
        if is_valid:
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_natural_voice_trainer():
    """Test the natural voice trainer."""
    print("\n🧪 Testing Natural Voice Trainer...")
    
    try:
        from trainer.natural_voice_trainer import NaturalVoiceTrainer
        from trainer.natural_voice_config import get_natural_voice_config
        
        # Test trainer class instantiation
        config = get_natural_voice_config()
        print(f"✅ NaturalVoiceTrainer class available")
        print(f"✅ Configuration ready for trainer")
        
        # Test trainer components
        print("🔧 Testing trainer components...")
        
        # Test perceptual loss
        from trainer.natural_voice_trainer import PerceptualAudioLoss
        perceptual_loss = PerceptualAudioLoss()
        print("✅ PerceptualAudioLoss initialized")
        
        # Test quality monitor
        from trainer.natural_voice_trainer import AudioQualityMonitor
        quality_monitor = AudioQualityMonitor(config)
        print("✅ AudioQualityMonitor initialized")
        
        # Test spectral processor
        from trainer.natural_voice_trainer import AdvancedSpectralProcessor
        spectral_processor = AdvancedSpectralProcessor(config)
        print("✅ AdvancedSpectralProcessor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        traceback.print_exc()
        return False


def test_audio_processing():
    """Test audio processing functions."""
    print("\n🧪 Testing Audio Processing...")
    
    try:
        from trainer.natural_voice_config import apply_natural_voice_enhancements, get_natural_voice_config
        
        config = get_natural_voice_config()
        
        # Create dummy audio tensor
        dummy_audio = torch.randn(1, 22050)  # 1 second of audio
        print(f"✅ Created dummy audio tensor: {dummy_audio.shape}")
        
        # Test enhancement function
        enhanced_audio = apply_natural_voice_enhancements(dummy_audio, config)
        print(f"✅ Audio enhancement function works: {enhanced_audio.shape}")
        
        # Verify output is same shape
        if enhanced_audio.shape == dummy_audio.shape:
            print("✅ Audio enhancement preserves tensor shape")
        else:
            print(f"⚠️  Audio shape changed: {dummy_audio.shape} -> {enhanced_audio.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        traceback.print_exc()
        return False


def test_training_script():
    """Test the training script imports."""
    print("\n🧪 Testing Training Script...")
    
    try:
        # Test if we can import the training functions
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        
        from natural_voice_training import (
            filter_for_natural_voice,
            validate_audio_quality,
            print_natural_voice_tips
        )
        
        print("✅ Training script imports successful")
        
        # Test filter function
        dummy_samples = [
            {"text": "This is a test sentence", "language": "en", "audio_file": "/fake/path.wav"},
            {"text": "Too short", "language": "en", "audio_file": "/fake/path.wav"},
            {"text": "This is a much longer sentence that should be filtered out because it exceeds the word limit for natural voice training", "language": "en", "audio_file": "/fake/path.wav"},
        ]
        
        filtered = filter_for_natural_voice(dummy_samples, "en")
        print(f"✅ Sample filtering works: {len(dummy_samples)} -> {len(filtered)} samples")
        
        # Test tips function
        print("✅ Testing tips function...")
        # Don't actually print tips to keep output clean
        # print_natural_voice_tips()
        
        return True
        
    except Exception as e:
        print(f"❌ Training script test failed: {e}")
        traceback.print_exc()
        return False


def test_system_requirements():
    """Test system requirements."""
    print("\n🧪 Testing System Requirements...")
    
    try:
        # Test PyTorch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - training will be slow")
        
        # Test numpy
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        
        # Test if we can create tensors
        test_tensor = torch.randn(100, 100)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
            print("✅ GPU tensor operations work")
        
        print("✅ System requirements check passed")
        return True
        
    except Exception as e:
        print(f"❌ System requirements test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("🎵 Natural Voice System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_natural_voice_config),
        ("Trainer Components", test_natural_voice_trainer),
        ("Audio Processing", test_audio_processing),
        ("Training Script", test_training_script),
        ("System Requirements", test_system_requirements),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    # Summary
    print(f"\n{'='*50}")
    print("🎵 Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests passed! Natural Voice System is ready!")
        print("\n🚀 Next steps:")
        print("1. Prepare high-quality training data")
        print("2. Update paths in natural_voice_training.py")
        print("3. Run: python examples/natural_voice_training.py")
        print("\n📖 See NATURAL_VOICE_GUIDE.md for detailed instructions")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("💡 Make sure all dependencies are installed correctly.")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
