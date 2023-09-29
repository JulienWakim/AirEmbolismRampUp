import numpy as np
import pyaudio
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('bmh')

SAMPLESIZE = 4096  # number of data points to read at a time
SAMPLERATE = 44100  # time resolution of the recording device (Hz)

p = pyaudio.PyAudio()  # instantiate PyAudio
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLERATE, input=True,
                frames_per_buffer=SAMPLESIZE)  # use default input device to open audio stream

# set up plotting
fig = plt.figure()
ax = plt.axes(xlim=(0, SAMPLESIZE - 1), ylim=(-9999, 9999))
line, = ax.plot([], [], lw=1)

# x axis data points
x = np.linspace(0, SAMPLESIZE - 1, SAMPLESIZE)


# methods for animation
def init():
    line.set_data([], [])
    return line,


def animate(i):
    y = np.frombuffer(stream.read(SAMPLESIZE), dtype=np.int16)
    line.set_data(x, y)
    return line,


FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

plt.show()

# stop and close the audio stream
stream.stop_stream()
stream.close()
p.terminate()
