#!/usr/bin/env python3
"""
Test GPTArgs Configuration
==========================
"""

import os
import sys

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig

def test_gpt_args():
    """Test GPTArgs configuration to check for model_dir attribute."""
    
    print("Testing GPTArgs configuration...")
    
    # Test basic GPTArgs creation
    model_args = GPTArgs(
        max_conditioning_length=220000,
        min_conditioning_length=66150,
        debug_loading_failures=True,
        max_wav_length=440000,
        max_text_length=350,
        mel_norm_file="/dummy/path/mel_stats.pth",
        dvae_checkpoint="/dummy/path/dvae.pth",
        xtts_checkpoint="/dummy/path/model.pth",
        tokenizer_file="/dummy/path/vocab.json",
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    
    print("✅ GPTArgs created successfully")
    
    # Check available attributes
    print("\nAvailable GPTArgs attributes:")
    for attr in dir(model_args):
        if not attr.startswith('_'):
            print(f"  - {attr}: {getattr(model_args, attr, 'N/A')}")
    
    # Check if model_dir exists
    if hasattr(model_args, 'model_dir'):
        print(f"\n✅ model_dir attribute exists: {model_args.model_dir}")
    else:
        print("\n❌ model_dir attribute not found")
        print("   This might be the source of the error")
    
    return model_args

if __name__ == "__main__":
    try:
        args = test_gpt_args()
        print("\n✅ GPTArgs test completed successfully")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
