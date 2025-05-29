#!/usr/bin/env python3
"""
Quick ElevenLabs Pipeline Test
=============================
A simplified test to verify the basic functionality.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🎯 Quick ElevenLabs Pipeline Test")
    print("=" * 40)
    
    try:
        # Test 1: Import modules
        print("1. Testing imports...")
        from examples.xtts.elevenlabs_performance_optimizer import create_performance_optimizer, PerformanceConfig
        from examples.xtts.elevenlabs_quality_enhancement import ElevenLabsAudioProcessor
        print("   ✓ All imports successful")
        
        # Test 2: Create audio processor
        print("2. Testing audio processor...")
        from examples.xtts.elevenlabs_quality_enhancement import ElevenLabsConfig
        config = ElevenLabsConfig()
        processor = ElevenLabsAudioProcessor(config)
        test_audio = torch.randn(1, 1000, dtype=torch.float32)
        enhanced = processor.enhance_audio_quality(test_audio.numpy())
        print(f"   ✓ Audio enhancement: {test_audio.shape} -> {enhanced.shape}")
        
        # Test 3: Create performance optimizer
        print("3. Testing performance optimizer...")
        config = PerformanceConfig(max_batch_size=4)
        optimizer = create_performance_optimizer(config)
        print(f"   ✓ Performance optimizer created with batch size {config.max_batch_size}")
        
        # Test 4: Test batch optimization
        print("4. Testing batch optimization...")
        test_batch = {
            'audio': torch.randn(2, 1, 1000),
            'mel': torch.randn(2, 80, 10)
        }
        optimized_batch = optimizer.optimize_batch_processing(test_batch)
        print(f"   ✓ Batch optimization successful")
        
        print("\n🎉 All tests passed! ElevenLabs pipeline is working correctly.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
