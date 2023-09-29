import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pyaudio
import wave

BASE_DIR = "./"  # Saves in current working directory (easiest way to do it)

def find_highest_note():
    audio_file = 'output.wav'
    y, sr = librosa.load(audio_file)

    # Get the pitches and times
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Find the highest pitch
    highest_pitch_idx = np.unravel_index(np.argmax(magnitudes, axis=None), magnitudes.shape)

    # Convert the pitch index to time
    highest_time = librosa.frames_to_time(highest_pitch_idx, sr=sr)

    print(f"The highest note occurs at {highest_time[0]} seconds.")



def plot():
    file = wave.open('output.wav', 'rb')

    sample_freq = file.getframerate()
    frames = file.getnframes()
    signal_wave = file.readframes(-1)

    file.close()

    time = frames / sample_freq

    # if one channel use int16, if 2 use int32
    audio_array = np.frombuffer(signal_wave, dtype=np.int16)

    times = np.linspace(0, time, num=frames)

    plt.figure(figsize=(15, 5))
    plt.plot(times, audio_array)
    plt.ylabel('Signal Wave')
    plt.xlabel('Time (s)')
    plt.xlim(0, time)
    plt.title('Recorded Sound')
    plt.show()

def record_and_display():
    # Set up the audio stream
    chunk = 4096  # This is the number of frames per buffer.
    sample_format = pyaudio.paInt16
    channels = 1  # Mono
    rate = 44100  # Standard sample rate for audio CDs
    seconds = 5

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    # x axis data points
    x = np.linspace(0, chunk - 1, chunk)

    print("Recording begins...")  # Begin Recording

    # Open an audio stream
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=rate,
                    frames_per_buffer=chunk,
                    output=True,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 5 seconds
    for i in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PortAudio interface
    p.terminate()

    print("Finished recording.")

    filepath = BASE_DIR + "output.wav"

    # Save the recorded data as a WAV file
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


record_and_display()
plot()
find_highest_note()
