#!/usr/bin/env python3
"""
Test GPTTrainer Initialization
==============================
"""

import os
import sys
import torch

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from xtts.shared_configs import BaseDatasetConfig

def test_gpt_trainer_init():
    """Test GPTTrainer initialization to check for potential errors."""
    
    print("Testing GPTTrainer initialization...")
    
    # Test basic configuration
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
    
    # Test audio config
    audio_config = XttsAudioConfig(
        sample_rate=22050, 
        dvae_sample_rate=22050, 
        output_sample_rate=24000
    )
    
    print("✅ XttsAudioConfig created successfully")
    
    # Test dataset config
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="test_dataset",
        path="/dummy/path",
        meta_file_train="/dummy/train.csv",
        meta_file_val="/dummy/eval.csv",
        language="es",
    )
    
    print("✅ BaseDatasetConfig created successfully")
    
    # Test trainer config (minimal)
    config = GPTTrainerConfig(
        epochs=1,
        output_path="/tmp/test_output",
        model_args=model_args,
        run_name="test_run",
        project_name="test_project",
        run_description="Test GPT XTTS training",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=1,
        eval_batch_size=1,
        datasets=[config_dataset],
        training_seed=54321,
        print_step=1,
        plot_step=1,
        save_step=1,
        optimizer="AdamW",
        lr=1e-6,
        test_sentences=[]
    )
    
    print("✅ GPTTrainerConfig created successfully")
    
    # Try to check what happens when we access model_args attributes through config
    try:
        print(f"Config model_args type: {type(config.model_args)}")
        print(f"Config model_args attributes: {[attr for attr in dir(config.model_args) if not attr.startswith('_')]}")
        
        # Check if there's any code trying to access model_dir
        if hasattr(config.model_args, 'model_dir'):
            print(f"model_dir found: {config.model_args.model_dir}")
        else:
            print("model_dir not found in model_args (this is expected)")
            
    except Exception as e:
        print(f"Error accessing model_args: {e}")
        import traceback
        traceback.print_exc()
    
    return config

if __name__ == "__main__":
    try:
        config = test_gpt_trainer_init()
        print("\n✅ GPTTrainer configuration test completed successfully")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
