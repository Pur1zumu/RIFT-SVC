# train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from schedulefree import AdamWScheduleFree

from model import CFM, DiT
from model.lightning_module import RIFTSVCLightningModule
from model.dataset import load_svc_dataset, collate_fn
from pytorch_lightning.callbacks import TQDMProgressBar
import time

class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.step_start_time = None
        self.total_steps = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.total_steps = trainer.max_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
        current_step = trainer.global_step
        total_steps = self.total_steps

        # Calculate elapsed time since training started
        elapsed_time = time.time() - self.start_time
        
        # Estimate average step time and remaining time
        average_step_time = elapsed_time / current_step if current_step > 0 else 0
        remaining_steps = total_steps - current_step
        remaining_time = average_step_time * remaining_steps if total_steps > 0 else 0

        # Format times to hh:mm:ss
        elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(remaining_time))

        # Update the progress bar with loss, elapsed time, remaining time, and remaining steps
        self.train_progress_bar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "elapsed_time": elapsed_time_str + "/" + remaining_time_str,
            "remaining_steps": str(remaining_steps) + "/" + str(total_steps)
        })


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    train_dataset = load_svc_dataset(
        data_dir=cfg.dataset.data_dir,
        meta_info_path=cfg.dataset.meta_info_path,
        max_frame_len=cfg.dataset.max_frame_len,
    )
    
    val_dataset = load_svc_dataset(
        data_dir=cfg.dataset.data_dir,
        meta_info_path=cfg.dataset.meta_info_path,
        max_frame_len=cfg.dataset.max_frame_len,
        split="test"
    )

    transformer = DiT(
        **cfg.model.cfg,
        num_speaker=train_dataset.num_speakers,
        mel_dim=cfg.dataset.n_mel_channels,
    )

    cfm = CFM(
        transformer=transformer,
        num_mel_channels=cfg.dataset.n_mel_channels,
    )

    warmup_steps = int(cfg.training.max_steps * cfg.training.warmup_ratio)
    optimizer = AdamWScheduleFree(
        cfm.parameters(),
        lr=cfg.training.learning_rate,
        betas=eval(cfg.training.betas),
        weight_decay=cfg.training.weight_decay,
        warmup_steps=warmup_steps,
    )
    model = RIFTSVCLightningModule(
        model=cfm,
        optimizer=optimizer,
        eval_sample_steps=cfg.training.eval_sample_steps,
        eval_cfg_strength=cfg.training.eval_cfg_strength,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_path,
        filename='model-{step}',
        save_top_k=-1,
        save_last='link',
        every_n_train_steps=cfg.training.save_per_steps,
    )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandbLogger(
        project=cfg.training.wandb_project,
        name=cfg.training.wandb_run_name,
        id=cfg.model.get('wandb_resume_id', None),
        resume='allow',
    )
    if wandb_logger.experiment.config:
        # Merge with existing config, giving priority to existing values
        wandb_logger.experiment.config.update(cfg_dict, allow_val_change=True)
    else:
        # If no existing config, set it directly
        wandb_logger.experiment.config.update(cfg_dict)


    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator='gpu',
        devices='auto',
        strategy='auto',
        precision='bf16-mixed',
        accumulate_grad_batches=cfg.training.grad_accumulation_steps,
        callbacks=[checkpoint_callback, CustomProgressBar()],
        logger=wandb_logger,
        val_check_interval=cfg.training.test_per_steps,
        check_val_every_n_epoch=None,
        gradient_clip_val=cfg.training.max_grad_norm,
        gradient_clip_algorithm='norm',
        log_every_n_steps=1,
    )


    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
        ),
    )

if __name__ == "__main__":
    main()