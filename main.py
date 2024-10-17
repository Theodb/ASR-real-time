import argparse
import os
import numpy as np
import speech_recognition as sr
import torch
import warnings

from datetime import datetime, timedelta
from queue import Queue
from time import sleep, time
from sys import platform
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from faster_whisper import WhisperModel #Cannot import it on my server problem with Cuda I am not avalaible to fix it
from datasets import load_dataset, Audio
import threading


def main():
    parser = argparse.ArgumentParser(description="Real-Time French Transcription with Whisper")
    parser.add_argument("--model", default="large-v3", help="Model to use",
                        choices=[
                            "tiny",
                            "base",
                            "small",
                            "medium",
                            "large",
                            "large-v3",
                            "distil-large-v2",
                            "distil-large-v3"
                        ])
    parser.add_argument("--language", default="fr", help="Language code for transcription (e.g., 'fr' for French).")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="Duration in seconds for each recording chunk.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="Seconds of silence to consider the end of a phrase.", type=float)
    parser.add_argument("--input_type", default="microphone", choices=["microphone", "dataset"], help="Input source: 'microphone' or 'dataset'.")
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Use 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # Define English-only models that require the '.en' suffix
    english_models = ["tiny", "base", "small", "medium", "large"]

    # Adjust model name based on language
    model = args.model
    if model in english_models and args.language == "en":
        model += ".en"

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        fp16 = True
        print("Using NVIDIA GPU with CUDA.")
    elif torch.backends.mps.is_available():
        device = "mps"
        fp16 = True  
        print("Using Apple Silicon MPS backend.")
    else:
        device = "cpu"
        fp16 = False
        print("Using CPU.")

    # Load models based on the device
    #if device == "cuda":
    #    # Use faster_whisper for NVIDIA GPUs
    #    print(f"Loading faster_whisper model '{model}'...")
    #    try:
    #        whisper_model = WhisperModel(model, device="cuda", compute_type="float16")
    #        # Set language to French and task to 'transcribe'
    #        language = args.language
    #    except Exception as e:
    #        print(f"Error loading faster_whisper model '{model}': {e}")
    #        return
    #else:
    # Use transformers' WhisperForConditionalGeneration for MPS and CPU
    print(f"Loading transformers Whisper model '{model}'...")
    try:
        model_path = f"distil-whisper/{model}" if "distil" in model else f"openai/whisper-{model}"
        processor = WhisperProcessor.from_pretrained(model_path, torch_dtype=torch.float16 if fp16 else torch.float32)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16 if fp16 else torch.float32)
        whisper_model.to(device)
        language = args.language
        # Prepare forced_decoder_ids for French transcription
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    except Exception as e:
        print(f"Error loading transformers Whisper model '{model}': {e}")
        return

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    # Initialize thread-safe queue
    data_queue = Queue()

    if args.input_type == 'microphone':
        # Initialize the recognizer
        recorder = sr.Recognizer()
        recorder.energy_threshold = args.energy_threshold
        recorder.dynamic_energy_threshold = False  # Prevent dynamic adjustments

        # Handle microphone selection for Linux users
        if 'linux' in platform:
            mic_name = args.default_microphone
            if mic_name.lower() == 'list':
                print("Available microphone devices:")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"{index}: {name}")
                return
            else:
                source = None
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name.lower() in name.lower():
                        source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
                if source is None:
                    print(f"Microphone with name containing '{mic_name}' not found.")
                    return
        else:
            source = sr.Microphone(sample_rate=16000)

        with source:
            recorder.adjust_for_ambient_noise(source)
            print("Adjusted for ambient noise.")

        def record_callback(_, audio: sr.AudioData) -> None:
            """
            Callback function to receive audio data when recordings finish.
            """
            data = audio.get_raw_data()
            data_queue.put(data)

        # Start listening in the background
        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    elif args.input_type == 'dataset':
        # Load the Fleurs French test dataset
        print("Loading Fleurs French test dataset...")
        fleurs = load_dataset("google/fleurs", "fr_fr", split="test")
        # Preprocess the dataset to resample audio to 16000 Hz
        fleurs = fleurs.cast_column("audio", Audio(sampling_rate=16000))
        # Initialize an iterator over the dataset
        dataset_iter = iter(fleurs)

        def simulate_dataset_input():
            """
            Function to simulate dataset input by feeding audio chunks into data_queue.
            """
            for sample in dataset_iter:
                audio = sample["audio"]["array"]
                # Normalize the audio data to int16
                audio_int16 = (audio * 32768).astype(np.int16)
                # Split audio into chunks of record_timeout duration
                chunk_size = int(16000 * record_timeout)  # Number of samples per chunk
                num_chunks = int(np.ceil(len(audio_int16) / chunk_size))
                for i in range(num_chunks):
                    chunk = audio_int16[i*chunk_size : (i+1)*chunk_size]
                    data = chunk.tobytes()
                    data_queue.put(data)
                    # Optionally sleep to simulate real-time (uncomment if needed)
                    # sleep(record_timeout)
            # Signal that dataset has been fully processed
            data_queue.put(None)

        # Start the simulation in a separate thread
        threading.Thread(target=simulate_dataset_input, daemon=True).start()
    else:
        print(f"Invalid input_type '{args.input_type}'.")
        return

    # Cue the user that we're ready to go.
    print("Model loaded.\nProcessing...")

    # Initialize phrase_time
    phrase_time = None
    last_sample = bytes()
    running = True

    while running:
        try:
            now = datetime.now()
            # Check if there's audio data in the queue
            if not data_queue.empty():
                data = data_queue.get()
                if data is None:
                    # No more data to process
                    running = False
                    continue
                # Determine if the phrase is complete based on silence
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    # Phrase is complete. Start a new one.
                    transcription.append('')
                # Update the last phrase time
                phrase_time = now

                # Append the new data to the buffer
                last_sample += data

                # Convert raw audio data to numpy array
                audio_np = np.frombuffer(last_sample, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe the audio based on the device
                #if device == "cuda":
                #    # Use faster_whisper for NVIDIA GPUs
                #    try:
                #        segments, info = whisper_model.transcribe(
                #            audio_np,
                #            beam_size=5,           # Beam size for a balance between speed and accuracy
                #            language=language,
                #            condition_on_previous_text=False
                #        )
                #        transcription_text = ''.join([segment.text for segment in segments]).strip()
                #    except Exception as e:
                #        warnings.warn(f"Faster Whisper transcription error: {e}")
                #        continue
                #else:
                # Use transformers for MPS and CPU
                try:
                    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
                    input_features = inputs.input_features.to(device).half()

                    # Generate transcription with forced_decoder_ids
                    with torch.no_grad():
                        predicted_ids = whisper_model.generate(
                            input_features,
                            forced_decoder_ids=forced_decoder_ids,
                            max_length=448,
                            do_sample=False,
                            num_beams=5  # Beam size for a balance between speed and accuracy
                        )
                    transcription_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                except Exception as e:
                    warnings.warn(f"Transformers Whisper transcription error: {e}")
                    continue

                # Update transcription
                transcription[-1] = transcription_text

                # Clear the console and display updated transcription
                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout
                print('', end='', flush=True)
            else:
                # Check if phrase_timeout has been reached
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    # Phrase is complete. Start a new one.
                    transcription.append('')
                    phrase_time = None
                    last_sample = bytes()
                # Prevent excessive CPU usage
                sleep(0.1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            warnings.warn(f"An unexpected error occurred: {e}")
            continue

    # Final transcription output
    print("\n\nFinal Transcription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()
