# RIFT-SVC: 基于矫正流变换器的歌声转换

<p align="center"><img src="./assets/logo.png" alt="RIFT-SVC" width="200"/></p>

**RIFT-SVC**是一种基于Rectified Flow Transformer的歌声转换模型实现。我们测试了几种相对于基础Diffusion Transformer的架构和训练改进。

**在线测试（[七海Nana7mi](https://space.bilibili.com/434334701)）：[https://huggingface.co/spaces/Pur1zumu/RIFT-SVC-Nanami](https://huggingface.co/spaces/Pur1zumu/RIFT-SVC-Nanami)**

## 新闻
- **2025-03-06: V3.0已发布。** 我们移除了V2中的whisper编码器，并添加了多个具有不同条件的无分类器引导。音色和发音都得到了改善。
- 2025-01-14: V2.0已发布。cfg-strength现在可以用来控制音色和发音之间的平衡。
- 2024-12-30: V1.0已发布。

---
## 环境准备

#### 1. 克隆仓库
```bash
git clone https://github.com/Pur1zumu/RIFT-SVC.git
cd RIFT-SVC
```

#### 2. 安装环境
* 需要安装uv：https://docs.astral.sh/uv/getting-started/installation/
```bash
uv sync
``` 

## 模型和数据准备

#### 3. 下载特征提取和声码器的预训练模型。
```bash
python pretrained/download.py   
```

#### 4. 下载用于微调的预训练权重。

| 模型 | 命令 |
| --- | --- |
| pretrain-v3_dit-512-8 | wget https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/pretrain-v3_dit-512-8.ckpt -O pretrained/pretrain-v3_dit-512-8.ckpt |
| pretrain-v3_dit-768-12 | wget https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/pretrain-v3_dit-768-12.ckpt -O pretrained/pretrain-v3_dit-768-12.ckpt |
| pretrain-v3_dit-1024-16 | wget https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/pretrain-v3_dit-1024-16.ckpt -O pretrained/pretrain-v3_dit-1024-16.ckpt |


#### 5. 准备数据并提取特征。
您应该这样组织您的数据：
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
`data/finetune`是微调的默认数据目录，但您可以更改为自己的数据目录。只需确保在其他命令和脚本中调整相应的参数。

音频文件默认应重采样为**44100 Hz**并对音量进行标准化至**-18 LUFS**。
运行：
```bash
python scripts/resample_normalize_audios.py --src data/finetune
```

准备元数据，运行：
```bash
python scripts/prepare_data_meta.py --data-dir $DATA_DIR
```
**参数：**
- `--data-dir`：您的数据目录路径。（必需）
- `--split-type`：数据分割类型：'random'或'stratified'。（默认：'random'）
- `--num-test`：用于'random'分割的测试样本数量。（默认：20）
- `--num-test-per-speaker`：用于'stratified'分割的每个说话者的测试样本数量。（默认：1）
- `--only-include-speakers`：要包含在元信息中的说话者的逗号分隔列表。（默认：无）
- `--seed`：用于可重复性的随机种子。（默认：42）


提取特征，运行：
```bash
python scripts/prepare_mel.py --data-dir $DATA_DIR --num-workers $NUM_WORKERS
python scripts/prepare_rms.py --data-dir $DATA_DIR --num-workers $NUM_WORKERS
python scripts/prepare_f0.py --data-dir $DATA_DIR --num-workers $NUM_WORKERS
python scripts/prepare_cvec.py --data-dir $DATA_DIR --num-workers $NUM_WORKERS
```
其中`$DATA_DIR`是您的数据目录路径（例如，`data/finetune`），`$NUM_WORKERS`是工作线程数。您可以根据GPU内存调整此值。


## 训练

#### 6. 开始微调

我们同时实现了Tensorboard和Wandb用于日志记录。

如果您想使用Wandb，您需要先登录Wandb：
```bash
wandb login
```
您可以在Wandb仪表板中找到训练日志。有关更多详情，请参见[Wandb](https://wandb.ai/)。

对于微调768-12模型，运行以下命令：
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

**参数：**
- `--config-name finetune`：`config/finetune.yaml`中的配置文件名称。
- `--model`：用于微调的模型架构。有关更多详情，请参见`config/model/`。模型应与预训练模型保持一致。
- `--training.run_name`：运行的名称。
- `--training.pretrained_path`：预训练模型的路径。
- `--training.learning_rate`：微调的学习率。
- `--training.max_steps`：微调的最大步数。
- `--training.weight_decay`：微调的权重衰减。
- `--training.batch_size_per_gpu`：每个GPU的批量大小。根据您的GPU内存调整此值。请注意，学习率也应相应调整。
- `--training.save_per_steps`：保存检查点的频率。
- `--training.test_per_steps`：测试的频率。
- `--training.time_schedule`：使用的噪声调度。默认值：`lognorm`。
- `--training.freeze_adaln_and_tembed`：是否冻结adaLN和时间嵌入。对于单一说话者训练，应将此设置为`true`以启用后续推理时的说话者音色增强。
- `--training.logger`：使用的日志记录器。默认值：`wandb`。如果您想使用Tensorboard，可以将其设置为`tensorboard`。然后使用`tensorboard --logdir logs`查看训练日志。

如果您想训练多个说话者，可以尝试上述的参数，也可以尝试解冻adaLN和时间嵌入：
```bash
+training.freeze_adaln_and_tembed=false
training.drop_spk_prob=0.2
```

##### VRAM消耗
下表显示了不同模型的VRAM消耗，所有测试均在一个3090 GPU上进行，使用`+training.freeze_adaln_and_tembed=true`。

| 模型 | 批量大小 | VRAM消耗 |
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

再次强调，您应相应调整学习率。
对于批量大小<=8，您可以考虑通过添加`--training.grad_accumulation_steps=n`来使用梯度累积，其中`n`是梯度累积步数。

##### 从检查点恢复
如果您希望从检查点恢复训练，可以添加以下参数：
```bash
[other arguments] +training.save_weights_only=False
```

然后，您可以通过添加以下参数从检查点恢复训练：
```bash
[other arguments] +training.resume_from_checkpoint=/path/to/checkpoint.ckpt +training.wandb_resume_id=your_wandb_run_id
```

## 推理

#### 7. 推理
基本推理命令：
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

高级推理命令：
```bash
python infer.py \
--model ckpts/finetune_ckpt-v3_dit-768-12_30000steps-lr0.00005/model-step=30000.ckpt \
--input 0.wav \
--output 0_steps32_cfg0.wav \
--speaker speaker1 \
--key-shift 0 \
--infer-steps 32 \
--batch-size 4 \
--ds-cfg-strength 0.1 \
--spk-cfg-strength 0.2 \
--skip-cfg-strength 0.1 \
--cfg-skip-layers 6 \
--cfg-rescale 0.7 \
--cvec-downsample-rate 2
```

**参数：**
- `--model`：微调模型的路径。
- `--input`：输入音频文件的路径。
- `--output`：输出音频文件的路径。
- `--speaker`：用于声音转换的说话者名称。
- `--key-shift`：以半音为单位的音高偏移（默认：0）。
- `--infer-steps`：推理步骤的数量（默认：32）。更高的值可能会产生更好的质量，但需要更长的时间。
- `--batch-size`: 并行推理的批处理大小（默认：1）。更高的值可以通过同时处理多个音频片段来加速推理，但需要更多的显存。
- `--ds-cfg-strength`：内容向量引导强度（默认：0.0）。更高的值可以改善内容保留和咬字清晰度。过高会用力过猛。我们建议初始试用值为0.1。
- `--spk-cfg-strength`：说话者引导强度（默认：0.0）。更高的值可以增强说话人相似度。过高可能导致音色失真。我们建议初始试用值为0.2-1。
- `--skip-cfg-strength`：跳层引导强度（实验性功能，默认：0.0）。增强指定层的特征渲染。效果取决于目标层的功能。我们建议初始试用值为0.1。
- `--cfg-skip-layers`：要跳过的目标层（实验性功能，默认：无）。我们建议初始试用值为（层数）/ 2，即12层模型为6。由于不同层具有不同功能，可以调整此值以找到最佳平衡。举例说明，如果某一层处理韵律相关特征，则跳过该层将使输出具有更多韵律特征。
- `--cfg-rescale`：无分类器引导重缩放因子（默认：0.7）。用于防止引导过饱和。
- `--cvec-downsample-rate`：用于反向引导的内容向量下采样率（默认：2）。

**修复声音中断和音高失真**

如果输出音频中遇到断音或破音，请尝试使用--robust-f0参数：

- `--robust-f0 0` 默认设置 - 无过滤，使用rmvpe f0提取器的原始输出
- `--robust-f0 1` 轻度过滤 - 有助于平滑轻微失真
- `--robust-f0 2` 激进过滤 - 提供最大平滑度，但可能会降低表现力

##### VRAM消耗

对于768-12模型，VRAM消耗约为~3GB。


##### GUI推理

要启动GUI网页推理，运行：

```bash
python gui_infer.py
```

要共享界面（创建公共URL），使用：

```bash
python gui_infer.py --share
```

