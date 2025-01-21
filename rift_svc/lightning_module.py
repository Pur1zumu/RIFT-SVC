import gc
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import wandb
import librosa

from pytorch_lightning import LightningModule

from rift_svc.metrics import mcd, psnr, si_snr, snr
from rift_svc.modules import get_mel_spectrogram
from rift_svc.nsf_hifigan import NsfHifiGAN
from rift_svc.utils import draw_mel_specs, l2_grad_norm


class RIFTSVCLightningModule(LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        cfg
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.eval_sample_steps = cfg['training']['eval_sample_steps']
        self.eval_cfg_strength = cfg['training']['eval_cfg_strength']
        self.log_media_per_steps = cfg['training']['log_media_per_steps']
        self.eval_spk_sim = cfg['training'].get('eval_spk_sim', False)

        self.vocoder = None
        self.spk_encoder = None
        self.save_hyperparameters(ignore=['model', 'optimizer', 'vocoder'])

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        mel_spec = batch['mel_spec']
        spk_id = batch['spk_id']
        f0 = batch['f0']
        rms = batch['rms']
        cvec = batch['cvec']
        whisper = batch['whisper']
        frame_lens = batch['frame_lens']

        loss, _ = self.model(
            mel_spec,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            whisper=whisper,
            lens=frame_lens,
        )

        self.log('train/loss', loss, prog_bar=True, logger=True)
        return loss
    
    def on_validation_start(self):
        self.optimizer.eval()
        if not self.trainer.is_global_zero:
            return

        if self.vocoder is None:
            self.vocoder =  NsfHifiGAN(
                'pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt').to(self.device)
        else:
            self.vocoder = self.vocoder.to(self.device)
        
        self.mcd = []
        self.si_snr = []
        self.psnr = []
        self.snr = []
        self.mse = []

        if self.eval_spk_sim:
            if self.spk_encoder is None:
                self.spk_encoder = torch.jit.load('pretrained/ecapa2.pt', map_location='cpu').to(self.device)
            else:
                self.spk_encoder = self.spk_encoder.to(self.device)

            self.spk_sim = []


    def on_validation_end(self, log=True):
        self.optimizer.train()
        if not self.trainer.is_global_zero:
            return

        if self.vocoder is not None:
            self.vocoder = self.vocoder.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        if self.spk_encoder is not None:
            self.spk_encoder = self.spk_encoder.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        
        metrics = {
            'val/mcd': np.mean(self.mcd),
            'val/si_snr': np.mean(self.si_snr),
            'val/psnr': np.mean(self.psnr),
            'val/snr': np.mean(self.snr),
            'val/mse': np.mean(self.mse)
        }

        if self.eval_spk_sim:
            metrics['val/spk_sim'] = np.mean(self.spk_sim)

        if log:
            self.logger.experiment.log(metrics, step=self.global_step)


    def validation_step(self, batch, batch_idx, log=True):
        if not self.trainer.is_global_zero:
            return
        
        global_step = self.global_step
        log_media_every_n_steps = self.log_media_every_n_steps

        spk_id = batch['spk_id']
        mel_gt = batch['mel_spec']
        rms = batch['rms']
        f0 = batch['f0']
        cvec = batch['cvec']
        whisper = batch['whisper']
        frame_lens = batch['frame_lens']

        mel_gen, _ = self.model.sample(
            src_mel=mel_gt,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            whisper=whisper,
            frame_lens=frame_lens,
            steps=self.eval_sample_steps,
            cfg_strength=self.eval_cfg_strength,
        )
        mel_gen = mel_gen.float()
        mel_gt = mel_gt.float()

        if self.eval_spk_sim:
            target_spk_id = torch.roll(spk_id, shifts=-1, dims=0)
            target_mel_gen, _ = self.model.sample(
                src_mel=mel_gt,
                spk_id=target_spk_id,
                f0=f0,
                rms=rms,
                cvec=cvec,
                whisper=whisper,
                frame_lens=frame_lens,
                steps=self.eval_sample_steps,
                cfg_strength=self.eval_cfg_strength,
            )
            target_mel_gen = target_mel_gen.float()

        for i in range(mel_gen.shape[0]):
            sample_idx = batch_idx * mel_gen.shape[0] + i
            wav_gen = self.vocoder(mel_gen[i:i+1, :frame_lens[i], :].transpose(1, 2), f0[i:i+1, :frame_lens[i]])
            wav_gt = self.vocoder(mel_gt[i:i+1, :frame_lens[i], :].transpose(1, 2), f0[i:i+1, :frame_lens[i]])

            wav_gen = wav_gen.squeeze(0)
            wav_gt = wav_gt.squeeze(0)

            if self.eval_spk_sim:
                target_wav_gen = self.vocoder(target_mel_gen[i:i+1, :frame_lens[i], :].transpose(1, 2), f0[i:i+1, :frame_lens[i]])
                target_wav_gen = target_wav_gen.squeeze(0)

                wav_sr = self.vocoder.h.sampling_rate

                # **Add Low-Pass Filtering Before Resampling using librosa**
                # Convert tensors to NumPy arrays
                target_wav_gen_np = target_wav_gen.float().cpu().numpy()
                wav_gt_np = wav_gt.float().cpu().numpy()

                # Apply low-pass filtering and resample to 16000 Hz
                target_wav_16khz_np = librosa.resample(target_wav_gen_np, orig_sr=wav_sr, target_sr=16000)
                wav_gt_16khz_np = librosa.resample(wav_gt_np, orig_sr=wav_sr, target_sr=16000)

                # Convert back to torch tensors
                target_wav_16khz = torch.from_numpy(target_wav_16khz_np).to(target_wav_gen.device)
                wav_gt_16khz = torch.from_numpy(wav_gt_16khz_np).to(wav_gt.device)

                spk_embs = self.spk_encoder(wav_gt_16khz)
                spk_embs = torch.roll(spk_embs, shifts=-1, dims=0)
                target_spk_embs = self.spk_encoder(target_wav_16khz)

                spk_sim = torch.cosine_similarity(spk_embs, target_spk_embs, dim=-1)
                self.spk_sim.append(spk_sim.cpu().item())

            mel_gen_i = get_mel_spectrogram(wav_gen).transpose(1, 2)
            mel_gt_i = get_mel_spectrogram(wav_gt).transpose(1, 2)

            mel_min, mel_max = self.model.mel_min, self.model.mel_max
            mel_gen_i = torch.clip(mel_gen_i, min=mel_min, max=mel_max)
            mel_gt_i = torch.clip(mel_gt_i, min=mel_min, max=mel_max)

            self.mcd.append(mcd(mel_gen_i, mel_gt_i).cpu().item())
            self.si_snr.append(si_snr(mel_gen_i, mel_gt_i).cpu().item())
            self.psnr.append(psnr(mel_gen_i, mel_gt_i).cpu().item())
            self.snr.append(snr(mel_gen_i, mel_gt_i).cpu().item())
            self.mse.append(F.mse_loss(mel_gen_i, mel_gt_i).cpu().item())

            if log:
                os.makedirs('.cache', exist_ok=True)
                if global_step % log_media_every_n_steps == 0:
                    torchaudio.save(f".cache/spk-{spk_id[i].item()}_{sample_idx}_gen.wav", wav_gen.cpu().to(torch.float32), 44100)
                    self.logger.experiment.log({
                        f"val-audio/spk-{spk_id[i].item()}_{sample_idx}-gen": wandb.Audio(f".cache/spk-{spk_id[i].item()}_{sample_idx}_gen.wav", sample_rate=44100),
                    }, step=self.global_step)

                    if self.eval_spk_sim:
                        torchaudio.save(f".cache/spk-{target_spk_id[i].item()}_{sample_idx}_converted.wav", target_wav_gen.cpu().to(torch.float32), 44100)
                        self.logger.experiment.log({
                            f"val-audio-converted/spk-{target_spk_id[i].item()}_{sample_idx}": wandb.Audio(f".cache/spk-{target_spk_id[i].item()}_{sample_idx}_converted.wav", sample_rate=44100)
                        }, step=self.global_step)
                
                if global_step == 0:
                    torchaudio.save(f".cache/spk-{spk_id[i].item()}_{sample_idx}_gt.wav", wav_gt.cpu().to(torch.float32), 44100)
                    self.logger.experiment.log({
                        f"val-audio/spk-{spk_id[i].item()}_{sample_idx}-gt": wandb.Audio(f".cache/spk-{spk_id[i].item()}_{sample_idx}_gt.wav", sample_rate=44100)
                    }, step=self.global_step)


                if global_step % log_media_every_n_steps == 0:
                    # Compute global min and max for consistent scaling across all plots
                    data_gt = mel_gt_i.squeeze().T.cpu().numpy()
                    data_gen = mel_gen_i.squeeze().T.cpu().numpy()
                    data_abs_diff = data_gen - data_gt

                    cache_path = f".cache/{sample_idx}_mel.jpg"
                    draw_mel_specs(data_gt, data_gen, data_abs_diff, cache_path)

                    self.logger.experiment.log({
                        f"val-mel/{sample_idx}_mel": wandb.Image(cache_path)
                    }, step=self.global_step)
    
    def on_test_start(self):
        self.on_validation_start()
    
    def on_test_end(self):
        self.on_validation_end(log=False)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, log=False)

    def on_before_optimizer_step(self, optimizer):
        # Calculate gradient norm
        norm = l2_grad_norm(self.model)

        self.log('train/grad_norm', norm, prog_bar=True, logger=True)

    @property
    def global_step(self):
        return self.trainer.global_step

    @property
    def log_media_every_n_steps(self):
        if self.log_media_per_steps is not None:
            return self.log_media_per_steps
        if self.save_every_n_steps is None:
            return self.trainer.val_check_interval
        return self.save_every_n_steps
    
    @property
    def save_every_n_steps(self):
        for callback in self.trainer.callbacks:
            if hasattr(callback, '_every_n_train_steps'):
                return callback._every_n_train_steps
        return None
    
    def state_dict(self, *args, **kwargs):
        # Temporarily store vocoder
        vocoder = self.vocoder
        self.vocoder = None
        
        # Get state dict without vocoder
        state = super().state_dict(*args, **kwargs)
        
        # Restore vocoder
        self.vocoder = vocoder
        return state