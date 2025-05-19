import os
import traceback

import torch

from xtts.gpt_trainer import train_gpt

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_model(
                language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length
            ):
                clear_gpu_cache()
                if not train_csv or not eval_csv:
                    return (
                        "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !",
                        "",
                        "",
                        "",
                        "",
                    )
                try:
                    # convert seconds to waveform frames
                    max_audio_length = int(max_audio_length * 22050)
                    config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
                        language,
                        num_epochs,
                        batch_size,
                        grad_acumm,
                        train_csv,
                        eval_csv,
                        output_path=output_path,
                        max_audio_length=max_audio_length,
                    )
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return (
                        f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}",
                        "",
                        "",
                        "",
                        "",
                    )

                # copy original files to avoid parameters changes issues
                os.system(f"cp {config_path} {exp_path}")
                os.system(f"cp {vocab_file} {exp_path}")

                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
                print("Model training done!")
                clear_gpu_cache()
                return "Model training done!", config_path, vocab_file, ft_xtts_checkpoint, speaker_wav