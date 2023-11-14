import torch
import librosa
import numpy as np
import pyaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Set up Whisper model and tokenizer
model_name = "facebook/wav2vec2-large-960h-lv60-self"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

# Set up audio input stream
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Continuously capture and transcribe audio
while True:
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)

    # Convert audio data to numpy array
    audio = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Preprocess audio and tokenize input
    input_values = tokenizer(audio, return_tensors="pt").input_values
    # Transcribe audio using Whisper model
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]

    # Display transcription
    print("Transcription: " + transcription)

# Clean up audio input stream
stream.stop_stream()
stream.close()
p.terminate()



