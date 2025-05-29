import gc
import os
import re

from trainer import Trainer, TrainerArgs

from shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager


def validate_text_quality(text, language="es"):
    """
    Validate text quality to prevent language mixing and pronunciation issues.
    
    Args:
        text (str): Text to validate
        language (str): Expected language code
    
    Returns:
        bool: True if text passes quality checks
    """
    if not text or len(text.strip()) < 3:
        return False
    
    # Check word count (avoid too short/long sentences)
    word_count = len(text.split())
    if not (5 <= word_count <= 50):
        return False
    
    # Basic character validation (remove obvious mixed-language content)
    # This is a simple check - you may want to add more sophisticated language detection
    
    # Check for excessive punctuation or special characters
    special_char_ratio = sum(1 for char in text if not char.isalnum() and not char.isspace()) / len(text)
    if special_char_ratio > 0.3:  # More than 30% special characters
        return False
    
    # Check for repeated characters (often indicates poor transcription)
    if re.search(r'(.)\1{3,}', text):  # 4 or more repeated characters
        return False
    
    # Check for mixed scripts (basic detection)
    has_latin = any('a' <= char.lower() <= 'z' for char in text)
    has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in text)
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    has_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
    
    # Count how many different scripts are present
    script_count = sum([has_latin, has_cyrillic, has_chinese, has_arabic])
    if script_count > 1:  # Mixed scripts detected
        return False
    
    return True


def improve_audio_quality_settings():
    """
    Return optimized audio settings for better TTS quality.
    
    Returns:
        dict: Audio configuration parameters
    """
    return {
        "sample_rate": 22050,
        "dvae_sample_rate": 22050,
        "output_sample_rate": 24000,
        # Enhanced STFT settings
        "fft_size": 2048,              # Better frequency resolution
        "win_length": 2048,            # Match fft_size  
        "hop_length": 512,             # Proportional adjustment
        # Improved silence handling (fixes word cutting)
        "trim_db": 30,                 # Less aggressive trimming
        "do_trim_silence": True,
        # Audio normalization (fixes robotic sounds)
        "do_sound_norm": True,         # Consistent volume levels
        "do_rms_norm": True,           # RMS normalization
        "db_level": -25.0,             # Appropriate level
        # Better synthesis quality
        "griffin_lim_iters": 100,      # More iterations for quality
        "power": 2.0,                  # Better reconstruction
        # Enhanced mel-spectrogram
        "num_mels": 100,               # Higher resolution
        "mel_fmin": 50.0,              # Better frequency range
        "mel_fmax": 8000.0,            # Appropriate upper limit
        "min_level_db": -120,          # Better dynamic range
        # Additional quality settings
        "preemphasis": 0.97,           # Slight preemphasis for clarity
        "ref_level_db": 20,
        "spec_gain": 20,
        "signal_norm": True,
        "symmetric_norm": True,
        "max_norm": 4.0,
        "clip_norm": True
    }


def train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length=255995, training_seed=54321):
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    OUT_PATH = os.path.join(output_path, "run", "training")

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = False  # for multi-gpu training
    START_WITH_EVAL = False  # if True it will start with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps

    # Define here the dataset that you want to use for the fine-tuning on.
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=train_csv,
        meta_file_val=eval_csv,
        language=language,
    )

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
    MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"
    XTTS_CONFIG_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))  # config.json file

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 files!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK, XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=220000,  # Increased from 132300 (10 secs instead of 6)
        min_conditioning_length=66150,   # Keep minimum conditioning
        debug_loading_failures=True,     # Enable debugging for loading failures
        max_wav_length=max_audio_length, # Use provided max length
        max_text_length=300,             # Increased from 200 to handle longer texts
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config with improved settings for better quality
    audio_config = XttsAudioConfig(
        sample_rate=22050, 
        dvae_sample_rate=22050, 
        output_sample_rate=24000,
        # Improved settings for audio quality
        fft_size=2048,              # Increased from 1024 for better frequency resolution
        win_length=2048,            # Match fft_size
        hop_length=512,             # Adjusted proportionally 
        trim_db=30,                 # Reduced from 45 to prevent cutting words
        do_sound_norm=True,         # Enable sound normalization
        do_rms_norm=True,           # Enable RMS normalization
        db_level=-25.0,             # Set appropriate RMS level
        griffin_lim_iters=100,      # Increased from 60 for better quality
        power=2.0,                  # Increased from 1.5 for better reconstruction
        num_mels=100,               # Increased from 80 for better resolution
        mel_fmin=50.0,              # Better for most voices
        mel_fmax=8000.0,            # Set appropriate upper frequency
        min_level_db=-120,          # Better dynamic range
        # Additional quality settings
        preemphasis=0.97,           # Slight preemphasis for clarity
        ref_level_db=20,
        spec_gain=20,
        signal_norm=True,
        symmetric_norm=True,
        max_norm=4.0,
        clip_norm=True
    )
    # training parameters config
    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="Improved GPT XTTS training for better audio quality",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        
        # Training settings - Improved
        batch_size=BATCH_SIZE,
        eval_batch_size=max(BATCH_SIZE // 2, 1),  # Smaller eval batch
        num_loader_workers=4,  # Reduced from 8 to avoid data loading issues
        training_seed=training_seed,
        
        # Eval settings - Improved  
        eval_split_max_size=512,  # Increased for better evaluation
        print_eval=True,  # Enable evaluation printing
        
        # Logging - More frequent for better monitoring
        print_step=25,    # More frequent printing
        plot_step=50,     # More frequent plotting  
        save_step=500,    # More frequent saving
        
        # Optimizer settings - Improved
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={
            "betas": [0.9, 0.999],   # Changed from [0.9, 0.96] for better stability
            "eps": 1e-8, 
            "weight_decay": 5e-3     # Reduced weight decay from 1e-2
        },
        lr=3e-06,  # Reduced learning rate from 5e-06 for more stable training
        lr_scheduler="CosineAnnealingLR",  # Changed from MultiStepLR for smoother decay
        lr_scheduler_params={"T_max": num_epochs * 1000, "eta_min": 1e-7},
        
        # Additional training stability improvements
        grad_clip=1.0,  # Add gradient clipping
        
        test_sentences=[]
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples with enhanced filtering
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    # Filter samples for better quality (avoid language mixing issues)
    def filter_samples(samples, language=language):
        """Filter training samples to improve quality and avoid language issues."""
        filtered = []
        rejected_reasons = {"too_short": 0, "too_long": 0, "poor_quality": 0, "mixed_language": 0}
        
        for sample in samples:
            text = sample.get("text", "")
            
            # Use the improved validation function
            if validate_text_quality(text, language):
                filtered.append(sample)
            else:
                # Count rejection reasons for debugging
                word_count = len(text.split())
                if word_count < 5:
                    rejected_reasons["too_short"] += 1
                elif word_count > 50:
                    rejected_reasons["too_long"] += 1
                else:
                    rejected_reasons["poor_quality"] += 1
        
        print(f"Sample filtering results: {rejected_reasons}")
        return filtered
    
    print(f"Original samples - Train: {len(train_samples)}, Eval: {len(eval_samples)}")
    train_samples = filter_samples(train_samples)
    eval_samples = filter_samples(eval_samples)
    print(f"Filtered samples - Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx = samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples
    gc.collect()

    return XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_ref
