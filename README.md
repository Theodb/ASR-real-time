# Real-Time Speech Transcription System

This repository contains a real-time speech transcription system using the Whisper model. The system is designed to transcribe speech with minimal delay and errors while a user is speaking. Although the real-time transcription functionality is operational, the evaluation metrics and multi-threaded capabilities are still under development.

## Table of Contents

- [Objectives](#objectives)
- [Current Status](#current-status)
- [Methodology](#methodology)
  - [Real-Time Transcription](#real-time-transcription)
  - [Challenges Faced](#challenges-faced)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Microphone Input](#microphone-input)
  - [Dataset Simulation](#dataset-simulation)
- [Project Structure](#project-structure)
- [References](#references)

## Current Status

- **Real-Time Transcription**: The system can transcribe speech in real-time from a microphone or simulated dataset input.  It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.
- **Evaluation Metrics**: is under development.
- **Multi-Threading**: Multi-threaded processing for multiple audio files is not yet implemented.
- **Challenges**:
  - Difficulty using the latest libraries on lab GPUs due to CUDA compatibility issues.
  - Difficulties to do quantization with the libraries of my labGPU
  - `faster_whisper` cannot be imported on the server; CUDA issues are pending resolution.
  - On macOS, a C++ implementation of Whisper exists and may be utilized in future updates.

## Methodology

### Real-Time Transcription

The system leverages the Whisper model for speech recognition, supporting both live microphone input and simulated dataset input for testing.

Key features:

- **Model Selection**: Supports various Whisper models.
- **Device Compatibility**: Automatically detects and utilizes available hardware (NVIDIA GPU with CUDA, Apple Silicon with MPS, or CPU).
- **Language Support**: Configurable language setting for transcription (default is French).
- **Input Sources**:
  - **Microphone**: Captures live audio for real-time transcription.
  - **Dataset Simulation**: Uses the French Fleurs dataset to simulate audio input, facilitating testing without live speech.

## Requirements

- Python 3.7 or higher
- Compatible hardware:
  - NVIDIA GPU with CUDA support (optional)
  - Apple Silicon (M1/M2) with MPS support (optional)
  - CPU (slower performance)
- Python packages listed in `requirements.txt`


## Usage

Run the real-time transcription script:

```bash
python main.py --model large-v3 --language fr --input_type microphone
```

**Arguments:**

- `--model`: Model name to use (default: `'large-v3'`). Choices include:
  - `tiny`, `base`, `small`, `medium`, `large`, `large-v3`
  - Distilled models: `distil-large-v2`, `distil-large-v3`
- `--language`: Language code for transcription (e.g., `'fr'` for French).
- `--input_type`: Input source, either `'microphone'` or `'dataset'`.

**Optional Arguments:**

- `--energy_threshold`: Energy level for the microphone to detect (default: `1000`).
- `--record_timeout`: Duration in seconds for each recording chunk (default: `2` seconds).
- `--phrase_timeout`: Seconds of silence to consider the end of a phrase (default: `3` seconds).
- `--default_microphone`: (Linux only) Default microphone name for `SpeechRecognition`.

### Dataset Simulation

To simulate input using the French Fleurs dataset:

```bash
python main.py --model large-v3 --language fr --input_type dataset
```

## References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [French Fleurs Dataset](https://huggingface.co/datasets/google/fleurs)

