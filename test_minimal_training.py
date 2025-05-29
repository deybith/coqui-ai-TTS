#!/usr/bin/env python3
"""
Minimal Training Test
====================
Test actual training initialization to verify no model_dir errors
"""

import gc
import os
import sys
import torch

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from trainer import Trainer, TrainerArgs
from xtts.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

def test_training_initialization():
    """Test if we can initialize training without model_dir errors."""
    
    print("🧪 Testing Training Initialization")
    print("=" * 50)
    
    # Use the test dataset paths
    train_csv = "/home/ubuntu/projects/coqui-ai-Trainer/data/test/dataset/metadata_train.csv"
    eval_csv = "/home/ubuntu/projects/coqui-ai-Trainer/data/test/dataset/metadata_eval.csv"
    output_path = "/home/ubuntu/projects/coqui-ai-Trainer/data/test_minimal_output"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    language = "es"
    num_epochs = 1
    batch_size = 1
    grad_acumm = 1
    max_audio_length = 255995
    
    # Configuration
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="minimal_test_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=train_csv,
        meta_file_val=eval_csv,
        language=language,
    )
    
    print("✅ Dataset config created")
    
    # XTTS setup
    OUT_PATH = os.path.join(output_path, "run", "training")
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)
    
    # File paths for XTTS model
    DVAE_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
    MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"
    TOKENIZER_FILE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"
    XTTS_CONFIG_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json"
    
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))
    
    print("✅ XTTS paths configured")
    
    # Model arguments - this is where the model_dir error might occur
    try:
        model_args = GPTArgs(
            max_conditioning_length=132300,
            min_conditioning_length=66150,
            debug_loading_failures=True,
            max_wav_length=max_audio_length,
            max_text_length=200,
            mel_norm_file=MEL_NORM_FILE,
            dvae_checkpoint=DVAE_CHECKPOINT,
            xtts_checkpoint=XTTS_CHECKPOINT,
            tokenizer_file=TOKENIZER_FILE,
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        )
        print("✅ GPTArgs created successfully")
    except Exception as e:
        print(f"❌ Error creating GPTArgs: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Audio configuration
    try:
        audio_config = XttsAudioConfig(
            sample_rate=22050, 
            dvae_sample_rate=22050, 
            output_sample_rate=24000
        )
        print("✅ XttsAudioConfig created successfully")
    except Exception as e:
        print(f"❌ Error creating XttsAudioConfig: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Training configuration
    try:
        config = GPTTrainerConfig(
            epochs=num_epochs,
            output_path=OUT_PATH,
            model_args=model_args,
            run_name="minimal_test",
            project_name="minimal_test_project",
            run_description="Minimal test training",
            dashboard_logger="tensorboard",
            audio=audio_config,
            batch_size=batch_size,
            eval_batch_size=batch_size,
            datasets=[config_dataset],
            training_seed=54321,
            print_step=1,
            plot_step=1,
            save_step=5,
            optimizer="AdamW",
            lr=1e-6,
            test_sentences=[]
        )
        print("✅ GPTTrainerConfig created successfully")
    except Exception as e:
        print(f"❌ Error creating GPTTrainerConfig: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try to initialize the model
    try:
        print("🔄 Attempting to initialize GPTTrainer...")
        model = GPTTrainer.init_from_config(config)
        print("✅ GPTTrainer initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing GPTTrainer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 SUCCESS! Training initialization completed without model_dir errors!")
    print("The configuration is valid and ready for training.")
    
    return True

if __name__ == "__main__":
    try:
        success = test_training_initialization()
        if success:
            print("\n✅ All tests passed - no model_dir attribute errors found!")
        else:
            print("\n❌ Tests failed - please check the errors above")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
