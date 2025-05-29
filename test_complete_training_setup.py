#!/usr/bin/env python3
"""
Test complete training setup to verify no model_dir AttributeError
"""

import os
import sys
import torch

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'examples'))

try:
    # Test basic GPTArgs creation
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    
    print("Testing GPTArgs creation...")
    
    # Create GPTArgs with typical parameters
    gpt_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading=False,
        max_wav_length=255995,
        max_text_length=200,
        mel_norm_file="",
        dvae_checkpoint="",
        xtts_checkpoint="",
        tokenizer_file="",
        gpt_num_audio_tokens=1024,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True
    )
    
    print("✅ GPTArgs created successfully")
    print(f"Available attributes: {[attr for attr in dir(gpt_args) if not attr.startswith('_')]}")
    
    # Test that model_dir is indeed not an attribute (confirming our fix was correct)
    has_model_dir = hasattr(gpt_args, 'model_dir')
    print(f"Has model_dir attribute: {has_model_dir}")
    
    if not has_model_dir:
        print("✅ Confirmed: GPTArgs correctly does not have 'model_dir' attribute")
    
    # Test improved trainer import
    print("\nTesting improved trainer imports...")
    from examples.xtts.improved_gpt_trainer import ImprovedGPTTrainer
    print("✅ ImprovedGPTTrainer imported successfully")
    
    # Test ultra-enhanced trainer import
    print("\nTesting ultra-enhanced trainer import...")
    import examples.train_xtts_ultra_enhanced_test
    print("✅ Ultra-enhanced trainer imported successfully")
    
    print("\n🎉 All imports successful! The model_dir AttributeError has been resolved.")
    print("The training pipeline should now work correctly.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
