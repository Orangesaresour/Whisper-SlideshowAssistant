import torch
import librosa
import numpy as np
import pyaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import cohere

co = cohere.Client('cTJFP4iCWKl8zsptu0wxNyX69KO42sUkfreZITmd') # This is your trial API key

# Set up Whisper model and tokenizer
model_name = "facebook/wav2vec2-large-960h-lv60-self"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

# Set up audio input stream
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

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
    response = co.generate(
        model='command',
        prompt='Please help me turn this weird semi-broken sentence into a coherent sentence with minimal changes. Just make adjustments to the misspellings or words in places that don\'t make sense contextually. Your output should just be a revised sentence that replaces the sentence I input for you. Your output should resemble the input sentence phonically and should resemble human speech without any spelling errors. Please also investigate text context as you correct the prompt and ensure your output is not significantly longer or shorter in word count than the input sentence length. Do not give me any extraneous output except for the revised sentence. I will repeat, never give me any other output such as "your text contains a trailing whitespace" or any other output aside from the rearranged sentence. Your sentence is: ' + transcription,
        max_tokens=300,
        temperature=0.9,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    print(response.generations[0].text)

# Clean up audio input stream
stream.stop_stream()
stream.close()
p.terminate()






