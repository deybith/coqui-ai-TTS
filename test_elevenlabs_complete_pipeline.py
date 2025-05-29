#!/usr/bin/env python3
"""
Complete ElevenLabs-Inspired Training Pipeline Test
==================================================

This script tests the entire enhanced training pipeline with:
- Audio quality enhancements
- Performance optimizations
- Memory management
- Comprehensive monitoring

Run this to validate that all ElevenLabs-inspired components work together.
"""

import os
import sys
import torch
import numpy as np
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our enhanced modules
from examples.xtts.elevenlabs_performance_optimizer import (
    PerformanceConfig, 
    create_performance_optimizer, 
    optimize_training_environment,
    get_optimal_config_for_gpu_memory
)
from examples.xtts.elevenlabs_quality_enhancement import ElevenLabsAudioProcessor, ElevenLabsConfig
from examples.xtts.elevenlabs_inspired_trainer import ElevenLabsInspiredTrainer


def create_test_data(batch_size: int = 4, seq_length: int = 1000):
    """Create synthetic test data for validation."""
    print("🔧 Creating synthetic test data...")
    
    # Create synthetic audio data
    audio_data = torch.randn(batch_size, 1, seq_length, dtype=torch.float32)
    mel_data = torch.randn(batch_size, 80, seq_length // 256, dtype=torch.float32)
    
    # Create test batch
    batch = {
        'audio': audio_data,
        'mel': mel_data,
        'tokens': torch.randint(0, 1000, (batch_size, 100)),
        'token_lens': torch.randint(50, 100, (batch_size,)),
        'audio_lens': torch.full((batch_size,), seq_length)
    }
    
    print(f"✓ Created test batch with {batch_size} samples")
    return batch


def test_audio_processor():
    """Test the ElevenLabsAudioProcessor."""
    print("\n🎵 Testing ElevenLabs Audio Processor...")
    
    try:
        # Create processor with config
        config = ElevenLabsConfig()
        processor = ElevenLabsAudioProcessor(config)
        
        # Test with synthetic audio
        test_audio = torch.randn(1, 22050, dtype=torch.float32)  # 1 second at 22kHz
        
        # Test enhancement
        enhanced_audio = processor.enhance_audio_quality(test_audio.numpy())
        
        print(f"✓ Audio enhancement successful")
        print(f"  Input shape: {test_audio.shape}")
        print(f"  Output shape: {enhanced_audio.shape}")
        print(f"  Enhancement applied: {not np.array_equal(test_audio.numpy(), enhanced_audio)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processor test failed: {e}")
        return False


def test_performance_optimizer():
    """Test the performance optimizer."""
    print("\n⚡ Testing Performance Optimizer...")
    
    try:
        # Test different GPU memory configurations
        configs = [
            ("Low-end GPU (8GB)", 8.0),
            ("Mid-range GPU (12GB)", 12.0),
            ("High-end GPU (24GB)", 24.0)
        ]
        
        for name, memory_gb in configs:
            config = get_optimal_config_for_gpu_memory(memory_gb)
            optimizer = create_performance_optimizer(config)
            
            print(f"✓ {name} configuration created")
            print(f"  Max batch size: {config.max_batch_size}")
            print(f"  Memory limit: {config.max_memory_usage_gb}GB")
            print(f"  Mixed precision: {config.use_mixed_precision}")
        
        # Test batch optimization
        test_batch = create_test_data(batch_size=4)
        optimized_batch = optimizer.optimize_batch_processing(test_batch)
        
        print(f"✓ Batch optimization successful")
        
        # Test memory tracking
        memory_stats = optimizer.memory_tracker.get_stats()
        print(f"✓ Memory tracking: {memory_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimizer test failed: {e}")
        return False


def test_enhanced_trainer():
    """Test the enhanced trainer with performance optimizations."""
    print("\n🚀 Testing Enhanced Trainer...")
    
    try:
        # Create performance config
        performance_config = PerformanceConfig(
            max_batch_size=8,
            enable_memory_optimization=True,
            use_mixed_precision=True,
            adaptive_batch_size=True
        )
        
        # Create dummy model for testing
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(80, 1)
                
            def forward(self, x):
                return self.linear(x.mean(dim=-1, keepdim=True))
        
        # Create dummy config
        class DummyConfig:
            def __init__(self):
                self.model_args = {}
                self.audio = {}
                self.batch_size = 4
                self.eval_batch_size = 2
                self.print_step = 1
                self.save_step = 10
                self.audio_len = 22050
        
        # Create dummy args for trainer
        class DummyArgs:
            def __init__(self):
                self.continue_path = None
                self.restore_path = None
                self.best_path = None
                
        # Create trainer
        config = DummyConfig()
        args = DummyArgs()
        trainer = ElevenLabsInspiredTrainer(
            args=args,
            config=config,
            output_path="/tmp/test_output",
            model=DummyModel(),
            performance_config=performance_config
        )
        
        print("✓ Enhanced trainer created successfully")
        
        # Test training step
        test_batch = create_test_data(batch_size=4, seq_length=config.audio_len)
        
        # Simulate training step
        trainer.model.train()
        losses = trainer.train_step(test_batch, step=1)
        
        print(f"✓ Training step completed")
        print(f"  Losses computed: {list(losses.keys())}")
        
        # Test quality report
        quality_report = trainer.get_quality_report()
        print(f"✓ Quality report generated: {len(quality_report)} metrics")
        
        # Test comprehensive status
        trainer.print_comprehensive_status()
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of all components."""
    print("\n🔗 Testing Component Integration...")
    
    try:
        # Initialize environment
        optimize_training_environment()
        
        # Create all components
        config = ElevenLabsConfig()
        audio_processor = ElevenLabsAudioProcessor(config)
        performance_config = PerformanceConfig(
            max_batch_size=4,
            enable_memory_optimization=True
        )
        
        # Test audio processing with performance optimization
        test_audio = torch.randn(2, 22050, dtype=torch.float32)
        enhanced_audio = audio_processor.enhance_audio_quality(test_audio.numpy())
        
        print(f"✓ Audio processing integrated")
        print(f"  Original audio energy: {test_audio.abs().mean():.4f}")
        print(f"  Enhanced audio energy: {torch.from_numpy(enhanced_audio).abs().mean():.4f}")
        
        # Test memory optimization
        if torch.cuda.is_available():
            test_audio_gpu = test_audio.cuda()
            enhanced_audio_gpu = audio_processor.enhance_audio_quality(test_audio_gpu.cpu().numpy())
            
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"✓ GPU integration successful (Memory used: {memory_used:.1f}MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def run_performance_benchmark():
    """Run a performance benchmark."""
    print("\n📊 Running Performance Benchmark...")
    
    try:
        # Create test data
        batch_sizes = [2, 4, 8]
        audio_lengths = [11025, 22050, 44100]  # 0.5s, 1s, 2s at 22kHz
        
        config = ElevenLabsConfig()
        audio_processor = ElevenLabsAudioProcessor(config)
        
        results = []
        
        for batch_size in batch_sizes:
            for audio_length in audio_lengths:
                # Create test audio
                test_audio = torch.randn(batch_size, audio_length, dtype=torch.float32)
                
                # Benchmark enhancement
                start_time = time.time()
                enhanced_audio = audio_processor.enhance_audio_quality(test_audio.numpy())
                end_time = time.time()
                
                processing_time = end_time - start_time
                samples_per_second = (batch_size * audio_length) / processing_time
                
                results.append({
                    'batch_size': batch_size,
                    'audio_length': audio_length,
                    'processing_time': processing_time,
                    'samples_per_second': samples_per_second
                })
                
                print(f"  Batch {batch_size}, Length {audio_length}: {processing_time:.3f}s ({samples_per_second:.0f} samples/s)")
        
        # Calculate averages
        avg_time = np.mean([r['processing_time'] for r in results])
        avg_throughput = np.mean([r['samples_per_second'] for r in results])
        
        print(f"✓ Benchmark completed")
        print(f"  Average processing time: {avg_time:.3f}s")
        print(f"  Average throughput: {avg_throughput:.0f} samples/s")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🎯 ElevenLabs-Inspired Training Pipeline Complete Test")
    print("=" * 60)
    
    # Check environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Run tests
    tests = [
        ("Audio Processor", test_audio_processor),
        ("Performance Optimizer", test_performance_optimizer),
        ("Enhanced Trainer", test_enhanced_trainer),
        ("Integration", test_integration),
        ("Performance Benchmark", run_performance_benchmark),
    ]
    
    results = []
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        try:
            success = test_func()
            results.append((test_name, success, time.time() - start_time))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False, time.time() - start_time))
    
    # Print summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for test_name, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<8} {test_name:<25} ({duration:.2f}s)")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    print(f"⏱️  Total time: {total_time:.2f}s")
    
    if failed == 0:
        print("\n🎉 All tests passed! ElevenLabs-inspired pipeline is ready for training.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
