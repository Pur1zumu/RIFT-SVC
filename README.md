# RIFT-SVC: Rectified Flow Transformer for Singing Voice Conversion

[中文文档](README_CN.md)

<p align="center"><img src="./assets/logo.png" alt="RIFT-SVC" width="200"/></p>

 Implementation of **RIFT-SVC**, a singing voice conversion model based on Rectified Flow Transformer. We tested several architecture-wise and training-wise improvements over vanilla Diffusion Transformer.

## News
- **2025-03-06: V3.0 is released.** We remove the V2's whisper encoder and add multiple classifier-free guidance with different conditions. Both articulation and timbre are improved.
- 2025-01-14: V2.0 is released. The cfg-strength can now be used to control the balance between timbre and articulation.
- 2024-12-30: V1.0 is released.

---
## Environment Preparation

#### 1. Clone the repository
```bash
git clone https://github.com/Pur1zumu/RIFT-SVC.git
cd RIFT-SVC
```

#### 2. Install environment
* uv is required: https://docs.astral.sh/uv/getting-started/installation/
```bash
uv sync
source .venv/bin/activate
``` 

## Models and Data Preparation

#### 3. Download pretrained models for feature extraction and vocoder.
```bash
python pretrained/download.py   
```

#### 4. Download pretrained weights for fine-tuning.

| Model | Command |
| --- | --- |
| pretrain-v3_dit-512-8 | wget https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/pretrain-v3_dit-512-8.ckpt -O pretrained/pretrain-v3_dit-512-8.ckpt |
| pretrain-v3_dit-768-12 | wget https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/pretrain-v3_dit-768-12.ckpt -O pretrained/pretrain-v3_dit-768-12.ckpt |
| pretrain-v3_dit-1024-16 | wget https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/pretrain-v3_dit-1024-16.ckpt -O pretrained/pretrain-v3_dit-1024-16.ckpt |


#### 5. Prepare data and extract features.
You should structure your data like this:
```
data/
    finetune/
        speaker1/
            audio1.wav
            audio2.wav
            ...
        speaker2/
            audio1.wav
            audio2.wav
            ...
```
`data/finetune` is the default data directory for fine-tuning, but you can change it to your own data directory. Just make sure to adjust the corresponding arguments in other commands and scripts.

The audio files are expected to be resampled to **44100 Hz** and loudness normalized to **-18 LUFS** by default. 
Run:
```bash
python scripts/resample_normalize_audios.py --src data/finetune
```

To prepare metadata, run:
```bash
python scripts/prepare_data_meta.py --data-dir $DATA_DIR
```
**Arguments:**
- `--data-dir`: The path to your data directory. (Required)
- `--split-type`: Type of data split: 'random' or 'stratified'. (Default: 'random')
- `--num-test`: Number of testing samples for 'random' split. (Default: 20)
- `--num-test-per-speaker`: Number of testing samples per speaker for 'stratified' split. (Default: 1)
- `--only-include-speakers`: Comma-separated list of speakers to include in the meta-information. (Default: None)
- `--seed`: Random seed for reproducibility. (Default: 42)


To extract features, run:
```bash
python scripts/prepare_mel.py --data-dir $DATA_DIR --num-workers $NUM_WORKERS
python scripts/prepare_rms.py --data-dir $DATA_DIR --num-workers $NUM_WORKERS
python scripts/prepare_f0.py --data-dir $DATA_DIR --num-workers $NUM_WORKERS
python scripts/prepare_cvec.py --data-dir $DATA_DIR --num-workers $NUM_WORKERS
```
where `$DATA_DIR` is the path to your data directory (e.g., `data/finetune`) and `$NUM_WORKERS` is the number of workers. You can adjust this value based on your GPU memory.


## Training

#### 6. Start Finetuning

We implement both Tensorboard and Wandb for logging.

If you want to use Wandb, you need to login to Wandb first:
```bash
wandb login
```
You can find the training logs in the Wandb dashboard. See [Wandb](https://wandb.ai/) for more details.

For finetuning 768-12 model, run the following command:
```bash
python train.py \
--config-name finetune \
model=dit-768-12 \
training.run_name=finetune_v3-dit-768-12_30000steps-lr0.00005 \
+training.pretrained_path=pretrained/pretrain-v3_dit-768-12.ckpt \
training.learning_rate=5e-5 \
training.weight_decay=0.01 \
training.max_steps=30000 \
training.batch_size_per_gpu=64 \
training.save_per_steps=1000 \
training.test_per_steps=1000 \
training.time_schedule=lognorm \
+training.freeze_adaln_and_tembed=true \
training.drop_spk_prob=0.0 \
training.logger=wandb or tensorboard
```

**Arguments:**
- `--config-name finetune`: The name of the config file in `config/finetune.yaml`.
- `--model`: The model architecture to use for fine-tuning. See `config/model/` for more details. The model should align with the pretrained model.
- `--training.run_name`: The name of the run.
- `--training.pretrained_path`: The path to the pretrained model.
- `--training.learning_rate`: The learning rate for fine-tuning.
- `--training.max_steps`: The maximum number of steps for fine-tuning.
- `--training.weight_decay`: The weight decay for fine-tuning.
- `--training.batch_size_per_gpu`: The batch size per GPU for fine-tuning. Adjust this value based on your GPU memory. Note that the learning rate should also be adjusted accordingly.
- `--training.save_per_steps`: The frequency of saving checkpoints.
- `--training.test_per_steps`: The frequency of testing.
- `--training.time_schedule`: The noise schedule to use. Default: `lognorm`.
- `--training.freeze_adaln_and_tembed`: Whether to freeze the adaLN and time embedding. For single speaker training, this should be set to `true` to enable the later inference-time speaker timbre enhancement.
- `--training.logger`: The logger to use. Default: `wandb`. If you want to use Tensorboard, you can set it to `tensorboard`. Then `tensorboard --logdir logs` to view the training logs.

If you want to train multiple speakers, you may try unfreeze the adaLN and time embedding:
```bash
+training.freeze_adaln_and_tembed=false
training.drop_spk_prob=0.2
```

##### VRAM Consumption
The below table shows the VRAM consumption of different models, all tested on one 3090 GPU, with `+training.freeze_adaln_and_tembed=true`.

| Model | Batch Size | VRAM Consumption |
| --- | --- | --- |
| v3_dit-512-8 | 64 | ~8GB |
| v3_dit-512-8 | 32 | ~5GB |
| v3_dit-512-8 | 16 | ~3GB |
| v3_dit-768-12 | 64 | ~17GB |
| v3_dit-768-12 | 32 | ~10GB |
| v3_dit-768-12 | 16 | ~6.5GB |
| v3_dit-768-12 | 8 | ~5GB |
| v3_dit-1024-16 | 32 | ~17GB |
| v3_dit-1024-16 | 16 | ~11GB |
| v3_dit-1024-16 | 8 | ~8.5GB |

Again, you should adjust the learning rate accordingly.
For batch size <=8, you may consider using gradient accumulation by adding `--training.grad_accumulation_steps=n`, where `n` is the number of gradient accumulation steps.

##### Resume from checkpoint
If you expect to resume training from a checkpoint, you can add the following argument:
```bash
[other arguments] +training.save_weights_only=False
```

Then, you can resume training from a checkpoint later by adding the following argument:
```bash
[other arguments] +training.resume_from_checkpoint=/path/to/checkpoint.ckpt +training.wandb_resume_id=your_wandb_run_id
```

## Inference

#### 7. Inference
Basic inference command:
```bash
python infer.py \
--model ckpts/finetune_ckpt-v3_dit-768-12_30000steps-lr0.00005/model-step=30000.ckpt \
--input 0.wav \
--output 0_steps32_cfg0.wav \
--speaker speaker1 \
--key-shift 0 \
--infer-steps 32 \
--batch-size 1
```

Advanced inference command:
```bash
python infer.py \
--model ckpts/finetune_ckpt-v3_dit-768-12_30000steps-lr0.00005/model-step=30000.ckpt \
--input 0.wav \
--output 0_steps32_cfg0.wav \
--speaker speaker1 \
--key-shift 0 \
--infer-steps 32 \
--batch-size 4 \
--ds-cfg-strength 0.2 \
--spk-cfg-strength 0.8 \
--skip-cfg-strength 0.0 \
--cfg-skip-layers 6 \
--cfg-rescale 0.7 \
--cvec-downsample-rate 2
```

**Arguments:**
- `--model`: The path to the fine-tuned model.
- `--input`: The path to the input audio file.
- `--output`: The path to the output audio file.
- `--speaker`: The speaker name for voice conversion.
- `--key-shift`: Pitch shift in semitones (default: 0).
- `--infer-steps`: The number of inference steps (default: 32). Higher values may produce better quality but take longer.
- `--batch-size`: Batch size for parallel inference (default: 1). Higher values can speed up inference by processing multiple segments simultaneously, but require more VRAM.
- `--ds-cfg-strength`: Downsampled content vector guidance strength (default: 0.2). Controls the emphasis on content fidelity.
- `--spk-cfg-strength`: Speaker guidance strength (default: 0.8). Higher values enhance speaker characteristics.
- `--skip-cfg-strength`: Skip layer guidance strength (default: 0.0). Affects how much the targeted intermediate layer's features are rendered on the output.
- `--cfg-skip-layers`: Layer to skip for classifier-free guidance (default: None). We recommend a initial trial value of (number of layers) / 2, which is 6 for 12-layer model. Since different layers have different functions, this value can be adjusted to find the best balance. For an illustration, if a layer processes prosody-related features, then skipping this layer will make the output has more prosody characteristics.
- `--cfg-rescale`: Classifier-free guidance rescale factor (default: 0.7). This is used to prevent over-saturation of the guidance [13].
- `--cvec-downsample-rate`: Downsampling rate for negative content vector creation (default: 2).

**Fixing Sound Breaks and Pitch Distortions**

If you experience sudden audio gaps during playback or abnormal pitch jumps in the output, try using the --robust-f0 argument:

- `--robust-f0 0` Default setting - no filtering, uses raw output from the rmvpe f0 extractor
- `--robust-f0 1` Light filtering - helps smooth minor distortions
- `--robust-f0 2` Aggressive filtering - provides maximum smoothing but may reduce expressiveness

##### VRAM Consumption

For 768-12 model, the VRAM consumption is ~3GB.


##### GUI Inference

To start the GUI application, run:

```bash
python gui_infer.py
```

For sharing the interface (creating a public URL), use:

```bash
python gui_infer.py --share
```

---

## Key Technical Details

### V3
#### Major Gains
- LogNorm noise scheduler[6]
- Combine multiple classifier-free guidance with different conditions, motivated by [12] (spetial thanks to [@KakaruHayate](https://github.com/KakaruHayate) for sharing the work)

#### Minor Gains
- DWConv's kernel size tuning

#### No Gains
- Remove time embedding, as in [11]
- Dropout

### V2
#### Major Gains
- Add whisper encoder[9] and use the hidden state from the second last layer to reduce timbre leakage, as explored in [10] [Removed in V3, mainly because of its timbre leakage and weak noise robustness]
- Add classifier-free guidance training by randomly dropping out the whisper embedding [Removed in V3]
- Add classifier-free guidance inference by f(cvec, null) + cfg_strength * (f(cvec, speaker) - f(cvec, null)) in each sample step
    - Note that this is different from the original classifier-free guidance in Diffusion Transformer, which is f(condition) + cfg_strength * (f(condition) - f(null)) [Removed in V3]
- Add post-LayerNorm at appropriate positions in the input embedding module

#### Minor Gains
- Add gating module for contentvec embedding and whisper embedding [Removed in V3]

### V1
#### Major Gains

- NSF-HIFIGAN[1] instead of BigVGAN[2]
- Add depth-wise conv in MLP, kind of like ConvNext block in 1d[3]
- Spectral parameterization (in terms of init and learning rate)[4]
- Scheduler-free optimizer (have not compared with vanilla AdamW, but no need to tune the scheduler anyway)[5] [Compared with AdamW in V3, which shows significantly faster convergence, but the final metrics show no much difference. Use this nevertheless because of better convergence.]

#### Minor Gains

- LogNorm noise scheduler[6] [Retested in V3 and shown major gains]
- Zero init for output blocks
- Classifier-free guidance training by randomly dropping out the speaker embedding (no quantitative test yet, but theoretically it should be better. But it not works for single speaker training.) [Removed in V2, add back in V3]
- Add an MLP for input condition embedding [Removed in V3]

#### No Gains

- UNet-like skip connections[7]
- MuP (I think it should be better, but I failed to pass the coordcheck. Now using Spectral parameterization instead)[8]

## References

[1] NSF-HIFIGAN: https://github.com/openvpi/SingingVocoders

[2] BigVGAN: https://github.com/NVIDIA/BigVGAN

[3] ConvNext: https://github.com/facebookresearch/ConvNeXt

[4] Spectral parameterization: Yang, G., Simon, J. B., & Bernstein, J. (2023). A spectral condition for feature learning. arXiv preprint arXiv:2310.17813.

[5] Scheduler-free optimizer: https://github.com/facebookresearch/schedule_free

[6] LogNorm noise scheduler: Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., ... & Rombach, R. (2024, March). Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning.

[7] UNet-like skip connections: Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., & Zhu, J. (2023). All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 22669-22679).

[8] MuP: Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., ... & Gao, J. (2022). Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466.

[9] Whisper encoder: https://huggingface.co/openai/whisper-large-v3

[10] Zhang, Li, et al. "Whisper-SV: Adapting Whisper for low-data-resource speaker verification." Speech Communication 163 (2024): 103103.

[11] Is Noise Conditioning Necessary for Denoising Generative Models?
https://arxiv.org/abs/2502.13129

[12] Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling
https://arxiv.org/abs/2411.18664

[12] Common Diffusion Noise Schedules and Sample Steps are Flawed
https://arxiv.org/abs/2305.08891
