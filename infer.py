import click
import librosa
import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast

from rift_svc import DiT, RF
from rift_svc.feature_extractors import HubertModelWithFinalProj, RMSExtractor, get_mel_spectrogram
from rift_svc.nsf_hifigan import NsfHifiGAN
from rift_svc.rmvpe import RMVPE
from rift_svc.utils import linear_interpolate_tensor, post_process_f0, f0_ensemble, f0_ensemble_light, get_f0_pw, get_f0_pm
from slicer import Slicer


torch.set_grad_enabled(False)


def extract_state_dict(ckpt):
    state_dict = ckpt['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_k = k.replace('model.', '')
            new_state_dict[new_k] = v
    spk2idx = ckpt['hyper_parameters']['cfg']['spk2idx']
    model_cfg = ckpt['hyper_parameters']['cfg']['model']
    dataset_cfg = ckpt['hyper_parameters']['cfg']['dataset']
    return new_state_dict, spk2idx, model_cfg, dataset_cfg


def load_models(model_path, device, use_fp16=True):
    """Load all required models and return them"""
    click.echo("Loading models...")
    
    ckpt = torch.load(model_path, map_location='cpu')
    state_dict, spk2idx, dit_cfg, dataset_cfg = extract_state_dict(ckpt)

    transformer = DiT(num_speaker=len(spk2idx), **dit_cfg)
    svc_model = RF(transformer=transformer)
    svc_model.load_state_dict(state_dict)
    svc_model = svc_model.to(device)
    
    if use_fp16 and device != 'cpu':
        svc_model = svc_model.half()
    
    svc_model.eval()
    
    vocoder = NsfHifiGAN('pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt').to(device)
    rmvpe = RMVPE(model_path="pretrained/rmvpe/model.pt", hop_length=160, device=device)
    hubert = HubertModelWithFinalProj.from_pretrained("pretrained/content-vec-best").to(device)
    rms_extractor = RMSExtractor().to(device)
    
    if use_fp16 and device != 'cpu':
        vocoder = vocoder.half()
        hubert = hubert.half()
        rms_extractor = rms_extractor.half()
    
    return svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg


def load_audio(file_path, target_sr):
    """Load and preprocess audio file"""
    click.echo("Loading audio...")
    audio, sr = torchaudio.load(file_path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    if len(audio.shape) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    return audio.numpy().squeeze()


def apply_fade(audio, fade_samples, fade_in=True):
    """Apply fade in/out using half of a Hanning window"""
    fade_window = np.hanning(fade_samples * 2)
    if fade_in:
        fade_curve = fade_window[:fade_samples]
    else:
        fade_curve = fade_window[fade_samples:]
    audio[:fade_samples] *= fade_curve
    return audio


def extract_features(audio_segment, sample_rate, hop_length, rmvpe, hubert, rms_extractor, 
                     device, key_shift=0, ds_cfg_strength=0.0, cvec_downsample_rate=2, target_loudness=-18.0,
                     robust_f0=0, use_fp16=True):
    """Extract all required features from an audio segment"""
    meter = pyln.Meter(sample_rate, block_size=0.1)
    original_loudness = meter.integrated_loudness(audio_segment)
    normalized_audio = pyln.normalize.loudness(audio_segment, original_loudness, target_loudness)

    max_amp = np.max(np.abs(normalized_audio))
    if max_amp > 1.0:
        normalized_audio = normalized_audio * (0.99 / max_amp)

    audio_tensor = torch.from_numpy(normalized_audio).float().unsqueeze(0).to(device)
    audio_16khz = torch.from_numpy(librosa.resample(normalized_audio, orig_sr=sample_rate, target_sr=16000)).float().unsqueeze(0).to(device)
    
    if use_fp16 and device.type != 'cpu':
        audio_tensor = audio_tensor.half()
        audio_16khz = audio_16khz.half()

    mel = get_mel_spectrogram(
        audio_tensor,
        sampling_rate=sample_rate,
        n_fft=2048,
        num_mels=128,
        hop_size=512,
        win_size=2048,
        fmin=40,
        fmax=16000
    ).transpose(1, 2)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with autocast(device_type=device_type, enabled=use_fp16):
        cvec = hubert(audio_16khz)["last_hidden_state"].squeeze(0)
    cvec = linear_interpolate_tensor(cvec, mel.shape[1])[None, :]

    if ds_cfg_strength > 0:
        cvec_ds = cvec.clone()
        cvec_ds = cvec_ds[0, ::2, :]
        cvec_ds = linear_interpolate_tensor(cvec_ds, cvec_ds.shape[0]//cvec_downsample_rate)
        cvec_ds = linear_interpolate_tensor(cvec_ds, mel.shape[1])[None, :]
    else:
        cvec_ds = None

    if robust_f0 > 0:
        time_step = hop_length / sample_rate
        f0_min = 40
        f0_max = 1100
        
        with autocast(device_type=device_type, enabled=use_fp16):
            rmvpe_f0 = rmvpe.infer_from_audio(audio_tensor, sample_rate=sample_rate, device=device)
        rmvpe_f0 = post_process_f0(rmvpe_f0, sample_rate, hop_length, mel.shape[1], silence_front=0.0, cut_last=False)
        pw_f0 = get_f0_pw(normalized_audio, sample_rate, time_step, f0_min, f0_max)
        pmac_f0 = get_f0_pm(normalized_audio, sample_rate, time_step, f0_min, f0_max)
        
        if robust_f0 == 1:
            with autocast(device_type=device_type, enabled=use_fp16):
                rms_np = rms_extractor(audio_tensor).squeeze().cpu().numpy()
            f0 = f0_ensemble_light(rmvpe_f0, pw_f0, pmac_f0, rms=rms_np)
        else:
            f0 = f0_ensemble(rmvpe_f0, pw_f0, pmac_f0)
    else:
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type, enabled=use_fp16):
            f0 = rmvpe.infer_from_audio(audio_tensor, sample_rate=sample_rate, device=device)
        f0 = post_process_f0(f0, sample_rate, hop_length, mel.shape[1], silence_front=0.0, cut_last=False)
    
    if key_shift != 0:
        f0 = f0 * 2 ** (key_shift / 12)
    f0 = torch.from_numpy(f0).float().to(device)[None, :]
    
    rms = rms_extractor(audio_tensor)
    
    return mel, cvec, cvec_ds, f0, rms, original_loudness


def run_inference(
    model, mel, cvec, f0, rms, cvec_ds, spk_id, 
    infer_steps, ds_cfg_strength, spk_cfg_strength, 
    skip_cfg_strength, cfg_skip_layers, cfg_rescale,
    frame_lengths=None, use_fp16=True
):
    """Run the actual inference through the model with optional batch processing"""
    device_type = 'cuda' if mel.device.type == 'cuda' else 'cpu'
    
    with autocast(device_type=device_type, enabled=use_fp16):
        mel_out, _ = model.sample(
            src_mel=mel,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            steps=infer_steps,
            bad_cvec=cvec_ds,
            ds_cfg_strength=ds_cfg_strength,
            spk_cfg_strength=spk_cfg_strength,
            skip_cfg_strength=skip_cfg_strength,
            cfg_skip_layers=cfg_skip_layers,
            cfg_rescale=cfg_rescale,
            frame_len=frame_lengths,
        )
    
    return mel_out


def generate_audio(vocoder, mel_out, f0, original_loudness=None, restore_loudness=True, use_fp16=True, expected_length=None):
    """Generate audio from mel spectrogram using vocoder"""
    device_type = 'cuda' if mel_out.device.type == 'cuda' else 'cpu'
    with autocast(device_type=device_type, enabled=use_fp16):
        audio_out = vocoder(mel_out.transpose(1, 2), f0)
    audio_out = audio_out.squeeze().cpu().numpy()

    if expected_length is not None:
        if len(audio_out) > expected_length:
            audio_out = audio_out[:expected_length]
        elif len(audio_out) < expected_length:
            audio_out = np.pad(audio_out, (0, expected_length - len(audio_out)), 'constant')

    if restore_loudness and original_loudness is not None:
        meter = pyln.Meter(44100, block_size=0.1)
        audio_out_loudness = meter.integrated_loudness(audio_out)
        audio_out = pyln.normalize.loudness(audio_out, audio_out_loudness, original_loudness)

        max_amp = np.max(np.abs(audio_out))
        if max_amp > 1.0:
            audio_out = audio_out * (0.99 / max_amp)
            
    return audio_out


def process_segment(
    audio_segment, 
    svc_model, vocoder, rmvpe, hubert, rms_extractor, 
    speaker_id, sample_rate, hop_length, device,
    key_shift=0, 
    infer_steps=32,
    ds_cfg_strength=0.0, 
    spk_cfg_strength=0.0, 
    skip_cfg_strength=0.0, 
    cfg_skip_layers=None, 
    cfg_rescale=0.7,
    cvec_downsample_rate=2,
    target_loudness=-18.0,
    restore_loudness=True,
    robust_f0=0,
    use_fp16=True
):
    """Process a single audio segment and return the converted audio"""
    mel, cvec, cvec_ds, f0, rms, original_loudness = extract_features(
        audio_segment, sample_rate, hop_length, rmvpe, hubert, rms_extractor, 
        device, key_shift, ds_cfg_strength, cvec_downsample_rate, target_loudness,
        robust_f0, use_fp16
    )
    
    spk_id = torch.LongTensor([speaker_id]).to(device)
    
    frame_length = torch.tensor([mel.shape[1]], device=device)
    
    mel_out = run_inference(
        model=svc_model, 
        mel=mel, 
        cvec=cvec, 
        f0=f0, 
        rms=rms, 
        cvec_ds=cvec_ds, 
        spk_id=spk_id,
        infer_steps=infer_steps,
        ds_cfg_strength=ds_cfg_strength,
        spk_cfg_strength=spk_cfg_strength,
        skip_cfg_strength=skip_cfg_strength,
        cfg_skip_layers=cfg_skip_layers,
        cfg_rescale=cfg_rescale,
        frame_lengths=frame_length,
        use_fp16=use_fp16
    )
    
    audio_out = generate_audio(
        vocoder, mel_out, f0, 
        original_loudness if restore_loudness else None, 
        restore_loudness, use_fp16,
        expected_length=len(audio_segment)
    )
    
    return audio_out


def batch_process_segments(
    segments_with_pos, 
    svc_model, vocoder, rmvpe, hubert, rms_extractor, 
    speaker_id, sample_rate, hop_length, device,
    key_shift=0, 
    infer_steps=32,
    ds_cfg_strength=0.0, 
    spk_cfg_strength=0.0, 
    skip_cfg_strength=0.0, 
    cfg_skip_layers=None, 
    cfg_rescale=0.7,
    cvec_downsample_rate=2,
    target_loudness=-18.0,
    restore_loudness=True,
    robust_f0=0,
    use_fp16=True,
    batch_size=1,
    gr_progress=None,
    progress_desc=None
):
    """Process audio segments in batches for faster inference"""
    if batch_size <= 1:
        results = []
        for i, (start_sample, chunk) in enumerate(tqdm(segments_with_pos, desc="Processing segments")):
            if gr_progress is not None:
                gr_progress(0.2 + (0.7 * (i / len(segments_with_pos))), desc=progress_desc.format(i+1, len(segments_with_pos)))
            audio_out = process_segment(
                chunk, svc_model, vocoder, rmvpe, hubert, rms_extractor,
                speaker_id, sample_rate, hop_length, device,
                key_shift, infer_steps, ds_cfg_strength, spk_cfg_strength,
                skip_cfg_strength, cfg_skip_layers, cfg_rescale,
                cvec_downsample_rate, target_loudness, restore_loudness,
                robust_f0, use_fp16
            )
            results.append((start_sample, audio_out, len(chunk)))
        return results
    
    sorted_with_idx = sorted(enumerate(segments_with_pos), key=lambda x: len(x[1][1]))
    sorted_segments = []
    original_indices = []
    
    for orig_idx, (pos, chunk) in sorted_with_idx:
        original_indices.append(orig_idx)
        sorted_segments.append((pos, chunk))

    batched_segments = [sorted_segments[i:i + batch_size] for i in range(0, len(sorted_segments), batch_size)]
    
    all_results = []
    
    for batch_idx, batch in enumerate(tqdm(batched_segments, desc="Processing batches")):
        if gr_progress is not None:
            gr_progress(
                0.2 + (0.7 * (batch_idx / len(batched_segments))),
                desc=progress_desc.format(batch_idx+1, len(batched_segments)))

        batch_start_samples = [pos for pos, _ in batch]
        batch_chunks = [chunk for _, chunk in batch]
        batch_lengths = [len(chunk) for chunk in batch_chunks]
        
        batch_features = []
        for chunk in batch_chunks:
            mel, cvec, cvec_ds, f0, rms, original_loudness = extract_features(
                chunk, sample_rate, hop_length, rmvpe, hubert, rms_extractor, 
                device, key_shift, ds_cfg_strength, cvec_downsample_rate, target_loudness,
                robust_f0, use_fp16
            )
            batch_features.append({
                'mel': mel, 
                'cvec': cvec, 
                'cvec_ds': cvec_ds, 
                'f0': f0, 
                'rms': rms, 
                'original_loudness': original_loudness,
                'length': mel.shape[1]
            })
        
        max_length = max(feat['length'] for feat in batch_features)
        
        padded_mels = []
        padded_cvecs = []
        padded_f0s = []
        padded_rmss = []
        frame_lengths = []
        original_loudness_values = []
        
        if ds_cfg_strength > 0:
            padded_cvec_ds = []
        
        for feat in batch_features:
            curr_len = feat['length']
            frame_lengths.append(curr_len)
            
            padded_mels.append(pad_tensor_to_length(feat['mel'], max_length))
            padded_cvecs.append(pad_tensor_to_length(feat['cvec'], max_length))
            padded_f0s.append(pad_tensor_to_length(feat['f0'], max_length))
            padded_rmss.append(pad_tensor_to_length(feat['rms'], max_length))
            
            if ds_cfg_strength > 0:
                padded_cvec_ds.append(pad_tensor_to_length(feat['cvec_ds'], max_length))
            
            original_loudness_values.append(feat['original_loudness'])
        
        batched_mel = torch.cat(padded_mels, dim=0)
        batched_cvec = torch.cat(padded_cvecs, dim=0)
        batched_f0 = torch.cat(padded_f0s, dim=0)
        batched_rms = torch.cat(padded_rmss, dim=0)
        
        if ds_cfg_strength > 0:
            batched_cvec_ds = torch.cat(padded_cvec_ds, dim=0)
        else:
            batched_cvec_ds = None
        
        frame_lengths = torch.tensor(frame_lengths, device=device)
        
        batch_spk_id = torch.LongTensor([speaker_id] * len(batch)).to(device)
        
        with torch.no_grad():
            mel_out = run_inference(
                model=svc_model,
                mel=batched_mel,
                cvec=batched_cvec,
                f0=batched_f0,
                rms=batched_rms,
                cvec_ds=batched_cvec_ds,
                spk_id=batch_spk_id,
                infer_steps=infer_steps,
                ds_cfg_strength=ds_cfg_strength,
                spk_cfg_strength=spk_cfg_strength,
                skip_cfg_strength=skip_cfg_strength,
                cfg_skip_layers=cfg_skip_layers,
                cfg_rescale=cfg_rescale,
                frame_lengths=frame_lengths,
                use_fp16=use_fp16
            )
            
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=use_fp16):
                audio_out = vocoder(mel_out.transpose(1, 2), batched_f0)
            
            for i in range(len(batch)):
                expected_audio_length = batch_lengths[i]
                
                curr_audio = audio_out[i].squeeze().cpu().numpy()
                
                if len(curr_audio) > expected_audio_length:
                    curr_audio = curr_audio[:expected_audio_length]
                elif len(curr_audio) < expected_audio_length:
                    curr_audio = np.pad(curr_audio, (0, expected_audio_length - len(curr_audio)), 'constant')
                
                if restore_loudness:
                    meter = pyln.Meter(44100, block_size=0.1)
                    curr_loudness = meter.integrated_loudness(curr_audio)
                    curr_audio = pyln.normalize.loudness(curr_audio, curr_loudness, original_loudness_values[i])
                    
                    max_amp = np.max(np.abs(curr_audio))
                    if max_amp > 1.0:
                        curr_audio = curr_audio * (0.99 / max_amp)
                
                expected_length = batch_lengths[i]
                
                all_results.append((batch_idx, i, batch_start_samples[i], curr_audio, expected_length, original_indices[batch_size * batch_idx + i]))
    
    all_results.sort(key=lambda x: x[5])
    
    return [(pos, audio, length) for _, _, pos, audio, length, _ in all_results]


def pad_tensor_to_length(tensor, length):
    """Pad a tensor to the specified length along the sequence dimension (dim=1)"""
    curr_len = tensor.shape[1]
    if curr_len >= length:
        return tensor
    
    pad_len = length - curr_len
    
    if tensor.dim() == 2:
        padding = (0, pad_len)
    elif tensor.dim() == 3:
        padding = (0, 0, 0, pad_len)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
    
    padded = torch.nn.functional.pad(tensor, padding, "constant", 0)
    return padded


@click.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to model checkpoint')
@click.option('--input', type=click.Path(exists=True), required=True, help='Input audio file')
@click.option('--output', type=click.Path(), required=True, help='Output audio file')
@click.option('--speaker', type=str, required=True, help='Target speaker')
@click.option('--key-shift', type=int, default=0, help='Pitch shift in semitones')
@click.option('--device', type=str, default=None, help='Device to use (cuda/cpu)')
@click.option('--infer-steps', type=int, default=32, help='Number of inference steps')
@click.option('--ds-cfg-strength', type=float, default=0.2, help='Downsampled content vector guidance strength')
@click.option('--spk-cfg-strength', type=float, default=0.8, help='Speaker guidance strength')
@click.option('--skip-cfg-strength', type=float, default=0.0, help='Skip layer guidance strength')
@click.option('--cfg-skip-layers', type=int, default=None, help='Layer to skip for classifier-free guidance')
@click.option('--cfg-rescale', type=float, default=0.7, help='Classifier-free guidance rescale factor')
@click.option('--cvec-downsample-rate', type=int, default=2, help='Downsampling rate for bad_cvec creation')
@click.option('--target-loudness', type=float, default=-18.0, help='Target loudness in LUFS for normalization')
@click.option('--restore-loudness', default=True, help='Restore loudness to original')
@click.option('--fade-duration', type=float, default=20.0, help='Fade duration in milliseconds')
@click.option('--robust-f0', type=int, default=0, help='Level of robust f0 filtering (0=none, 1=light, 2=aggressive)')
@click.option('--slicer-threshold', type=float, default=-30.0, help='Threshold for audio slicing in dB')
@click.option('--slicer-min-length', type=int, default=3000, help='Minimum length of audio segments in milliseconds')
@click.option('--slicer-min-interval', type=int, default=100, help='Minimum interval between audio segments in milliseconds')
@click.option('--slicer-hop-size', type=int, default=10, help='Hop size for audio slicing in milliseconds')
@click.option('--slicer-max-sil-kept', type=int, default=200, help='Maximum silence kept in milliseconds')
@click.option('--use-fp16', is_flag=True, default=True, help='Use float16 precision for faster inference')
@click.option('--batch-size', type=int, default=1, help='Batch size for parallel inference')
def main(
    model,
    input,
    output,
    speaker,
    key_shift,
    device,
    infer_steps,
    ds_cfg_strength,
    spk_cfg_strength,
    skip_cfg_strength,
    cfg_skip_layers,
    cfg_rescale,
    cvec_downsample_rate,
    target_loudness,
    restore_loudness,
    fade_duration,
    robust_f0,
    slicer_threshold,
    slicer_min_length,
    slicer_min_interval,
    slicer_hop_size,
    slicer_max_sil_kept,
    use_fp16,
    batch_size
):
    """Convert the voice in an audio file to a target speaker."""

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg = load_models(model, device, use_fp16)

    try:
        speaker_id = spk2idx[speaker]
    except KeyError:
        raise ValueError(f"Speaker {speaker} not found in the model's speaker list, valid speakers are {spk2idx.keys()}")
    
    hop_length = 512
    sample_rate = 44100

    audio = load_audio(input, sample_rate)

    slicer = Slicer(
        sr=sample_rate,
        threshold=slicer_threshold,
        min_length=slicer_min_length,
        min_interval=slicer_min_interval,
        hop_size=slicer_hop_size,
        max_sil_kept=slicer_max_sil_kept
    )

    click.echo("Slicing audio...")
    segments_with_pos = slicer.slice(audio)

    if restore_loudness:
        click.echo(f"Will restore loudness to original")

    fade_samples = int(fade_duration * sample_rate / 1000)

    click.echo(f"Processing {len(segments_with_pos)} segments with batch size {batch_size}...")
    
    with torch.no_grad():
        processed_segments = batch_process_segments(
            segments_with_pos, svc_model, vocoder, rmvpe, hubert, rms_extractor,
            speaker_id, sample_rate, hop_length, device,
            key_shift, infer_steps, ds_cfg_strength, spk_cfg_strength,
            skip_cfg_strength, cfg_skip_layers, cfg_rescale,
            cvec_downsample_rate, target_loudness, restore_loudness,
            robust_f0, use_fp16, batch_size
        )

    result_audio = np.zeros(len(audio) + fade_samples)
    
    for idx, (start_sample, audio_out, expected_length) in enumerate(processed_segments):
        if len(audio_out) > expected_length:
            audio_out = audio_out[:expected_length]
        elif len(audio_out) < expected_length:
            audio_out = np.pad(audio_out, (0, expected_length - len(audio_out)), 'constant')
        
        if idx > 0:
            audio_out = apply_fade(audio_out.copy(), fade_samples, fade_in=True)
            result_audio[start_sample:start_sample + fade_samples] *= \
                np.linspace(1, 0, fade_samples)
        
        if idx < len(processed_segments) - 1:
            audio_out[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        result_audio[start_sample:start_sample + len(audio_out)] += audio_out

    result_audio = result_audio[:len(audio)]

    click.echo("Saving output...")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output, torch.from_numpy(result_audio).unsqueeze(0), sample_rate)
    click.echo("Done!")


if __name__ == '__main__':
    main()