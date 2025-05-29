#!/usr/bin/env python3
"""
TTS Audio Quality Enhancement Selector
=====================================

This script helps you choose the right level of audio quality enhancements
based on your needs and available computational resources.
"""

import os
import psutil
import torch


def check_system_resources():
    """Check available system resources."""
    print("🔍 Checking system resources...")
    
    # CPU info
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # GPU info
    gpu_available = torch.cuda.is_available()
    gpu_memory_gb = 0
    if gpu_available:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"  💻 CPU cores: {cpu_count}")
    print(f"  🧠 RAM: {memory_gb:.1f} GB")
    if gpu_available:
        print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  🎮 GPU memory: {gpu_memory_gb:.1f} GB")
    else:
        print("  🎮 GPU: Not available")
    
    return {
        'cpu_count': cpu_count,
        'memory_gb': memory_gb,
        'gpu_available': gpu_available,
        'gpu_memory_gb': gpu_memory_gb
    }


def analyze_dataset(train_csv, eval_csv):
    """Analyze dataset characteristics."""
    print("\\n📊 Analyzing dataset...")
    
    try:
        import pandas as pd
        
        # Read datasets
        train_df = pd.read_csv(train_csv)
        eval_df = pd.read_csv(eval_csv)
        
        # Basic stats
        train_size = len(train_df)
        eval_size = len(eval_df)
        
        # Text length analysis
        if 'text' in train_df.columns:
            text_lengths = train_df['text'].str.split().str.len()
            avg_text_length = text_lengths.mean()
            max_text_length = text_lengths.max()
        else:
            avg_text_length = "Unknown"
            max_text_length = "Unknown"
        
        print(f"  📝 Training samples: {train_size}")
        print(f"  📝 Evaluation samples: {eval_size}")
        print(f"  📏 Average text length: {avg_text_length}")
        print(f"  📏 Maximum text length: {max_text_length}")
        
        return {
            'train_size': train_size,
            'eval_size': eval_size,
            'avg_text_length': avg_text_length,
            'max_text_length': max_text_length
        }
        
    except Exception as e:
        print(f"  ⚠️  Could not analyze dataset: {e}")
        return None


def recommend_training_level(resources, dataset_info=None):
    """Recommend the best training level based on resources and dataset."""
    print("\\n🎯 Analyzing optimal training configuration...")
    
    score = 0
    recommendations = []
    
    # Resource scoring
    if resources['memory_gb'] >= 16:
        score += 2
    elif resources['memory_gb'] >= 8:
        score += 1
    
    if resources['gpu_available'] and resources['gpu_memory_gb'] >= 8:
        score += 3
    elif resources['gpu_available'] and resources['gpu_memory_gb'] >= 4:
        score += 2
    elif resources['gpu_available']:
        score += 1
    
    if resources['cpu_count'] >= 8:
        score += 1
    
    # Dataset considerations
    if dataset_info:
        if dataset_info['train_size'] > 1000:
            score += 1
        if isinstance(dataset_info['avg_text_length'], (int, float)) and dataset_info['avg_text_length'] > 20:
            score += 1
    
    # Make recommendation
    if score >= 7:
        level = "ultra"
        title = "🌟 Ultra Enhanced Training"
        script = "examples/train_xtts_ultra_enhanced.py"
        description = "Maximum quality with all cutting-edge optimizations"
        features = [
            "✨ Perceptual loss optimization",
            "🎯 Multi-resolution spectral analysis", 
            "📈 Dynamic warmup scheduling",
            "🎚️ Adaptive audio processing",
            "🔍 Real-time quality monitoring"
        ]
        requirements = "High-end system (16GB+ RAM, 8GB+ GPU)"
        
    elif score >= 4:
        level = "improved"
        title = "🚀 Improved Enhanced Training"
        script = "examples/train_xtts_improved.py"
        description = "Excellent quality with enhanced optimizations"
        features = [
            "🔇 Audio cut-off prevention",
            "🤖 Robotic sound elimination",
            "🌍 Language consistency",
            "📊 Advanced data filtering",
            "⚡ Training stability"
        ]
        requirements = "Mid-range system (8GB+ RAM, 4GB+ GPU)"
        
    else:
        level = "standard"
        title = "✅ Standard Enhanced Training"
        script = "examples/train_xtts.py (with auto-improvements)"
        description = "Great quality with automatic optimizations"
        features = [
            "🔧 All basic improvements applied automatically",
            "🎵 Better audio processing",
            "📈 Improved training stability",
            "🔍 Basic quality monitoring"
        ]
        requirements = "Basic system (4GB+ RAM, any GPU or CPU)"
    
    return {
        'level': level,
        'title': title,
        'script': script,
        'description': description,
        'features': features,
        'requirements': requirements,
        'score': score
    }


def generate_training_command(recommendation, custom_params=None):
    """Generate the appropriate training command."""
    
    if recommendation['level'] == "ultra":
        base_config = {
            'language': 'es',
            'num_epochs': 20,
            'batch_size': 6,
            'grad_acumm': 3,
            'max_audio_length': 22  # seconds
        }
    elif recommendation['level'] == "improved": 
        base_config = {
            'language': 'es',
            'num_epochs': 15,
            'batch_size': 8,
            'grad_acumm': 2,
            'max_audio_length': 20  # seconds
        }
    else:  # standard
        base_config = {
            'language': 'es',
            'num_epochs': 10,
            'batch_size': 16,
            'grad_acumm': 1,
            'max_audio_length': 30  # seconds
        }
    
    # Apply custom parameters if provided
    if custom_params:
        base_config.update(custom_params)
    
    return base_config


def main():
    """Main function to run the enhancement selector."""
    
    print("🎵 TTS Audio Quality Enhancement Selector")
    print("=" * 55)
    print()
    print("This tool will help you choose the optimal training configuration")
    print("for your system and dataset to achieve the best audio quality.")
    print()
    
    # Check system resources
    resources = check_system_resources()
    
    # Get dataset paths
    print("\\n📁 Dataset Configuration:")
    train_csv = input("  Enter path to training CSV: ").strip()
    eval_csv = input("  Enter path to evaluation CSV: ").strip()
    
    # Analyze dataset if paths provided
    dataset_info = None
    if train_csv and eval_csv and os.path.exists(train_csv) and os.path.exists(eval_csv):
        dataset_info = analyze_dataset(train_csv, eval_csv)
    
    # Get recommendation
    recommendation = recommend_training_level(resources, dataset_info)
    
    # Display recommendation
    print("\\n" + "="*60)
    print(f"🎯 RECOMMENDED: {recommendation['title']}")
    print("="*60)
    print(f"📝 {recommendation['description']}")
    print(f"💻 {recommendation['requirements']}")
    print()
    print("🔧 Features included:")
    for feature in recommendation['features']:
        print(f"  {feature}")
    print()
    print(f"🚀 Script to run: {recommendation['script']}")
    
    # Generate configuration
    print("\\n⚙️  Recommended Configuration:")
    config = generate_training_command(recommendation)
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Custom configuration option
    print("\\n🔧 Would you like to customize the configuration? (y/n): ", end="")
    customize = input().strip().lower()
    
    if customize in ['y', 'yes']:
        print("\\nCustomize parameters (press Enter to keep default):")
        custom_params = {}
        
        for key, default in config.items():
            response = input(f"  {key} [{default}]: ").strip()
            if response:
                # Try to convert to appropriate type
                try:
                    if isinstance(default, int):
                        custom_params[key] = int(response)
                    elif isinstance(default, float):
                        custom_params[key] = float(response)
                    else:
                        custom_params[key] = response
                except ValueError:
                    print(f"    Warning: Invalid value for {key}, using default")
        
        if custom_params:
            config.update(custom_params)
            print("\\n✅ Updated configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
    
    # Generate final instructions
    print("\\n" + "="*60)
    print("🚀 NEXT STEPS")
    print("="*60)
    
    if recommendation['level'] == "ultra":
        print("1. Edit examples/train_xtts_ultra_enhanced.py with your parameters:")
    elif recommendation['level'] == "improved":
        print("1. Edit examples/train_xtts_improved.py with your parameters:")
    else:
        print("1. Edit examples/train_xtts.py with your parameters:")
    
    for key, value in config.items():
        print(f"   {key} = {repr(value)}")
    
    print(f"\\n2. Run the training script:")
    print(f"   python {recommendation['script']}")
    
    print("\\n3. Monitor training progress:")
    print("   - Watch for smooth loss curves")
    print("   - Listen to generated samples regularly")
    print("   - Check for quality warnings in logs")
    
    print("\\n4. Troubleshooting resources:")
    print("   - COMPLETE_AUDIO_ENHANCEMENT_GUIDE.md - Full guide")
    print("   - QUICK_FIX_GUIDE.md - Quick reference")
    print("   - TROUBLESHOOTING.md - Detailed troubleshooting")
    
    print("\\n🎵 Happy training! You should see significant audio quality improvements.")


if __name__ == "__main__":
    main()
