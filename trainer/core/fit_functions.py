import gc
import logging
import os
import sys
import traceback
from contextlib import suppress
from typing import Any

import torch
import torch.distributed as dist

from trainer.generic_utils import (
    KeepAverage,
    remove_experiment_folder,
)
from trainer.io import (
    save_best_model,
    save_checkpoint,
)
from trainer.logging import DummyLogger
from trainer.utils.cuda_memory import cuda_meminfo, should_reduce_batch_size
from trainer.utils.distributed import (
    rank_zero_only,
)

logger = logging.getLogger("trainer")

class FitFunctions:

    ###################################
    # FIT FUNCTIONS
    ###################################

    def _fit(self) -> None:
        """🏃 train -> evaluate -> test for the number of epochs."""
        self._restore_best_loss()

        self.total_steps_done = self.restore_step

        for epoch in range(self.config.epochs):
            if self.num_gpus > 1:
                # let all processes sync up before starting with a new epoch of training
                dist.barrier()
            self.callbacks.on_epoch_start(self)
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage() if self.config.run_eval else None
            self.epochs_done = epoch
            self.c_logger.print_epoch_start(epoch, self.config.epochs, self.output_path)
            if not self.skip_train_epoch and not self.start_with_eval:
                self.train_epoch()
            if self.config.run_eval:
                self.eval_epoch()
            if epoch >= self.config.test_delay_epochs and self.args.rank <= 0:
                self.test_run()

            self.c_logger.print_epoch_end(
                epoch,
                self.keep_avg_eval.avg_values if self.config.run_eval else self.keep_avg_train.avg_values,  # type: ignore[union-attr]
            )
            if self.args.rank in [None, 0]:
                self.save_best_model()
            self.callbacks.on_epoch_end(self)
            self.start_with_eval = False

    def fit_with_largest_batch_size(self, starting_batch_size: int = 2048) -> None:
        """Find and use the largest possible batch size for training.
        
        Args:
            starting_batch_size (int): Initial batch size to try. Defaults to 2048.
        """
        cuda_meminfo()
        bs = starting_batch_size
        
        def clear_memory():
            """Clear GPU memory and garbage collection."""
            gc.collect()
            torch.cuda.empty_cache()
            
        while True:
            clear_memory()
            try:
                self.config.batch_size = bs
                logger.info(" > current batch size: %i", self.config.batch_size)
                self._fit()
                break
            except (RuntimeError, Exception) as exception:  # catches RuntimeError and torch.cuda.OutOfMemoryError
                if bs > 1 and should_reduce_batch_size(exception):
                    bs //= 2
                    clear_memory()
                else:
                    raise

    def fit(self) -> None:
        """Start the training process.
        
        This method handles the main training loop, including error handling and cleanup.
        It manages:
        - Training execution
        - Keyboard interrupts (with optional checkpoint saving)
        - Distributed training cleanup
        - Dashboard logger finalization
        - Error logging and cleanup
        
        Raises:
            RuntimeError: If there's an error during training
            torch.cuda.OutOfMemoryError: If GPU runs out of memory
            Exception: For any other unexpected errors
        """
        try:
            self._fit()
            if self.args.rank == 0:
                self.dashboard_logger.finish()
        except KeyboardInterrupt:
            logger.info(" > Keyboard interrupt detected.")
            self._handle_interrupt()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(" > Training error: %s", str(e))
            remove_experiment_folder(self.output_path)
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(" > Unexpected error: %s", str(e))
            remove_experiment_folder(self.output_path)
            traceback.print_exc()
            sys.exit(1)
            
    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt with graceful shutdown."""
        if self.config.save_on_interrupt:
            logger.info(" > Saving model before exiting...")
            self.save_checkpoint()
            self.update_training_dashboard_logger()
            
        # call the keyboard interrupt callback
        self.callbacks.on_keyboard_interrupt(self)
        
        # cleanup
        remove_experiment_folder(self.output_path)
        if self.num_gpus > 1:
            dist.destroy_process_group()
        if self.args.rank == 0:
            self.dashboard_logger.finish()
            
        # exit gracefully
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)  # pylint: disable=protected-access

    def profile_fit(
        self, torch_profiler: torch.profiler.profile, epochs: int | None = None, small_run: int | None = None
    ) -> torch.profiler.profile:
        """Run training under the PyTorch profiler to analyze performance.

        This method configures and runs the training process with profiling enabled,
        allowing detailed analysis of CPU, GPU, and memory usage.

        Args:
            torch_profiler (torch.profiler.profile): Configured PyTorch profiler instance
            epochs (int, optional): Number of epochs to profile. If None, uses config value
            small_run (int, optional): Number of samples to use for profiling. If None, uses full dataset

        Returns:
            torch.profiler.profile: The profiler instance with collected data

        Example::
            Profile CPU, GPU and memory usage with Tensorboard logging:

            >>> import torch
            >>> profiler = torch.profiler.profile(
            >>>     activities=[
            >>>         torch.profiler.ProfilerActivity.CPU,
            >>>         torch.profiler.ProfilerActivity.CUDA,
            >>>     ],
            >>>     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            >>>     on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler/"),
            >>>     record_shapes=True,
            >>>     profile_memory=True,
            >>>     with_stack=True,
            >>> )
            >>> prof = trainer.profile_fit(profiler, epochs=1, small_run=64)
        """
        # Configure profiling environment
        self._setup_profiling_env(torch_profiler, epochs, small_run)
        
        try:
            # Run training with profiler
            self.torch_profiler.start()
            self.fit()
            self.torch_profiler.stop()
        except Exception as e:
            logger.error(" > Profiling error: %s", str(e))
            self.torch_profiler.stop()
            raise
            
        return self.torch_profiler
        
    def _setup_profiling_env(
        self, torch_profiler: torch.profiler.profile, epochs: int | None, small_run: int | None
    ) -> None:
        """Configure the environment for profiling.
        
        Args:
            torch_profiler: The PyTorch profiler instance
            epochs: Number of epochs to profile
            small_run: Number of samples for profiling
        """
        # Use dummy logger to avoid overhead
        self.dashboard_logger = DummyLogger()
        
        # Configure training parameters
        if epochs:
            self.config.epochs = epochs
        if small_run:
            self.setup_small_run(small_run)
            
        # Disable eval and testing to focus on training
        self.config.run_eval = False
        self.config.test_delay_epochs = 9999999
        
        # Setup profiler callbacks and instance
        self.callbacks_on_train_step_end = [  # pylint: disable=attribute-defined-outside-init
            lambda trainer: trainer.torch_profiler.step()
        ]
        self.torch_profiler = torch_profiler  # pylint: disable=attribute-defined-outside-init

    @rank_zero_only
    def save_best_model(self) -> None:
        """Save the best model. It only saves if the current target loss is smaller then the previous."""
        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train) or float("inf")

        # save the model and update the best_loss
        self.best_loss = save_best_model(
            {"train_loss": train_loss, "eval_loss": eval_loss},
            self.best_loss,
            self.config,
            self._get_model(),
            self.optimizer,
            self.scaler if self.use_amp_scaler else None,
            self.total_steps_done,
            self.epochs_done,
            self.output_path,
            keep_all_best=self.config.save_all_best,
            keep_after=self.config.save_best_after,
            save_func=self.dashboard_logger.save_model,
        )

    @rank_zero_only
    def save_checkpoint(self) -> None:
        """Save the current model checkpoint."""
        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train)

        save_checkpoint(
            self.config,
            self._get_model(),
            self.optimizer,
            self.scaler if self.use_amp_scaler else None,
            self.total_steps_done,
            self.epochs_done,
            self.output_path,
            model_loss={"train_loss": train_loss, "eval_loss": eval_loss},
            save_n_checkpoints=self.config.save_n_checkpoints,
            save_func=self.dashboard_logger.save_model,
        )

    @rank_zero_only
    def update_training_dashboard_logger(
        self, batch: dict[str, Any] | list[Any] | None = None, outputs: dict[str, Any] | None = None
    ) -> None:
        aliases = [
            f"epoch-{self.epochs_done}",
            f"step-{self.total_steps_done}",
        ]
        self.dashboard_logger.add_artifact(
            file_or_dir=self.output_path, name="checkpoint", artifact_type="model", aliases=aliases
        )

        # training visualizations
        if batch is not None and outputs is not None:
            model = self._get_model()
            with suppress(NotImplementedError):
                model.train_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
