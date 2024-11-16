#!/usr/bin/env python3
"""
generate_contentvec.py

This script reads a meta-information JSON file containing speakers and their corresponding audio files,
extracts content vectors for each audio file using a HuBERT-like model, and saves the content vectors
as .contentvec.pt files in the same directory as the original audio files.

Usage:
    python generate_contentvec.py --meta-info META_INFO_JSON --data-dir DATA_DIR --model-path MODEL_PATH [OPTIONS]

Options:
    --meta-info FILE_PATH        Path to the meta_info.json file. (Required)
    --data-dir DIRECTORY         Path to the root of the preprocessed dataset directory. (Required)
    --model-path FILE_PATH       Path to the pre-trained HuBERT-like model file. (Required)
    --sample-rate INTEGER        Sample rate of the audio files in Hz. (Default: 16000)
    --verbose                    Enable verbose output.
"""

import json
import os
import sys
from pathlib import Path
from multiprocessing import Process, Queue, current_process
import multiprocessing
from torch import multiprocessing as mp

import torch
import torch.nn as nn
import torchaudio
import click
from tqdm import tqdm
import numpy as np

# Import the HuBERT-like model
# Adjust the import based on your project structure
from transformers import HubertModel

# Define the model class, including any necessary modifications
class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def worker_process(audio_subset, data_dir, model_path, sample_rate, queue, verbose, device_id=None):
    """
    Worker function to extract content vectors from a subset of audio files.

    Args:
        audio_subset (list): List of audio entries to process.
        data_dir (Path): Root directory of the preprocessed dataset.
        model_path (str): Path to the pre-trained HuBERT-like model.
        sample_rate (int): Sample rate of the audio files in Hz.
        queue (Queue): Multiprocessing queue to communicate progress.
        verbose (bool): If True, enable verbose output.
        device_id (int or None): CUDA device ID to use. If None, use CPU.
    """
    device = torch.device(f'cuda:{device_id}' if device_id is not None and torch.cuda.is_available() else 'cpu')
    try:
        # Load model configuration and initialize the model
        model = HubertModelWithFinalProj.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
    except Exception as e:
        queue.put(f"Error initializing model in process {current_process().name}: {e}")
        return

    # Disable gradient computation
    torch.set_grad_enabled(False)

    for audio in audio_subset:
        speaker = audio.get('speaker')
        file_name = audio.get('file_name')

        if not speaker or not file_name:
            if verbose:
                queue.put(f"Skipping invalid entry: {audio} in process {current_process().name}")
            continue

        # Construct paths
        wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
        contentvec_path = Path(data_dir) / speaker / f"{file_name}.cvec.pt"

        if not wav_path.is_file():
            if verbose:
                queue.put(f"Warning: WAV file not found: {wav_path} in process {current_process().name}")
            continue

        try:
            # Load audio
            waveform, sr = torchaudio.load(str(wav_path))
            waveform = waveform.to(device)

            # Ensure waveform has proper shape for RMVPE (batch, samples)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)  # Shape: (1, samples)
            elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
                # Convert to mono by averaging channels
                waveform = waveform.mean(dim=0, keepdim=True)  # Shape: (1, samples)

            # Resample if necessary (assuming preprocessing handled sample rate)
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate).to(device)
                waveform = resampler(waveform)

            # Run the model
            with torch.no_grad():
                contentvec = model(waveform)  # Shape: (1, seq_len, hidden_size)
                contentvec = contentvec["last_hidden_state"].squeeze(0).cpu()  # Remove batch dimension

            # Save the content vector
            torch.save(contentvec, contentvec_path)

            if verbose:
                queue.put(f"Saved content vector: {contentvec_path} in process {current_process().name}")

            # Send progress update
            queue.put("PROGRESS")

        except Exception as e:
            queue.put(f"Error processing {wav_path} in process {current_process().name}: {e}")
            continue

    # Notify completion of this process
    queue.put(f"Process {current_process().name} completed.")

@click.command()
@click.option(
    '--meta-info',
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
    help='Path to the meta_info.json file.'
)
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Path to the root of the preprocessed dataset directory.'
)
@click.option(
    '--model-path',
    type=click.Path(exists=True, file_okay=True, readable=True),
    default='pretrained/content-vec-best',
    show_default=True,
    help='Path to the pre-trained HuBERT-like model file.'
)
@click.option(
    '--sample-rate',
    type=int,
    default=16000,
    show_default=True,
    help='Sample rate of the audio files in Hz.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def generate_contentvec(meta_info, data_dir, model_path, sample_rate, verbose):
    """
    Generate content vectors for each audio file specified in the meta_info.json and save them as .contentvec.pt files.
    """
    # Move the start method setting here as a fallback
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
        
    # Load meta_info.json
    try:
        with open(meta_info, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        click.echo(f"Error reading meta_info.json: {e}", err=True)
        sys.exit(1)

    speakers = meta.get('speakers', [])
    train_audios = meta.get('train_audios', [])
    test_audios = meta.get('test_audios', [])

    # Combine train and test audios
    all_audios = train_audios + test_audios

    if not all_audios:
        click.echo("No audio files found in meta_info.json.", err=True)
        sys.exit(1)

    # Detect number of available CUDA devices
    num_devices = torch.cuda.device_count()
    if num_devices > 1:
        devices = list(range(num_devices))
    elif num_devices == 1:
        devices = [0]
    else:
        devices = []

    if verbose:
        click.echo(f"Number of CUDA devices available: {num_devices}")

    if devices:
        click.echo(f"Using CUDA devices: {devices}")
    else:
        click.echo("No CUDA devices available. Using CPU.")

    # Split audios among devices
    num_processes = len(devices) if devices else 1

    split_audios = [[] for _ in range(num_processes)]
    for i, audio in enumerate(all_audios):
        split_audios[i % num_processes].append(audio)

    # Create a multiprocessing Queue for communication
    queue = Queue()

    # Create and start processes
    processes = []
    for i in range(num_processes):
        device_id = devices[i] if devices else None
        p = Process(
            target=worker_process,
            args=(
                split_audios[i],
                Path(data_dir),
                model_path,
                sample_rate,
                queue,
                verbose,
                device_id
            ),
            name=f"Process-{i}"
        )
        p.start()
        processes.append(p)

    # Initialize tqdm progress bar
    with tqdm(total=len(all_audios), desc="Extracting Content Vectors", unit="file") as pbar:
        completed_processes = 0
        while completed_processes < num_processes:
            message = queue.get()
            if message == "PROGRESS":
                pbar.update(1)
            elif message.startswith("Saved content vector") and verbose:
                pbar.set_postfix({"Last Saved": message})
            elif message.startswith("Warning") and verbose:
                pbar.write(message)
            elif message.startswith("Error"):
                pbar.write(message)
            elif message.startswith("Process") and "completed" in message:
                completed_processes += 1
                if verbose:
                    pbar.write(message)
            else:
                if verbose:
                    pbar.write(message)

    # Ensure all processes have finished
    for p in processes:
        p.join()

    click.echo("Content vector extraction complete.")

if __name__ == "__main__":
    # Set start method to spawn
    mp.set_start_method('spawn', force=True)

    generate_contentvec()
