# Python standard modules
import time
import threading as th
import pyaudio as pa
import librosa
import matplotlib.pyplot as plt

#
# Pylab
# (Numpy, Scipy, Matplotlib)
import scipy.signal
import numpy as np


class Audiooo:

    def __init__(self, sr=22050):
        self.sr = sr
        self.buff_size = 1024
        self.buff = None
        self.specgram = np.ndarray((self.buff_size // 2 + 1, 0))

        self.pa = pa.PyAudio()
        self.input_stream = pa.Stream(PA_manager=self.pa, input=True, rate=self.sr, channels=1, format=pa.paInt16)
        self.output_stream = pa.Stream(PA_manager=self.pa, output=True, rate=self.sr, channels=1, format=pa.paFloat32)

        self.chirp = None
        self.get_data = True

    def get_audio_data(self):
        dt = np.dtype('int16')
        dt = dt.newbyteorder('>')

        while self.get_data is True:
            # read data
            _buff = self.input_stream.read(self.buff_size)
            self.buff = np.frombuffer(_buff, dtype=dt).astype('float32').copy() // 32767

            # process data
            fft = np.fft.fft(self.buff)[0:self.buff_size // 2 + 1]
            fft = fft[:, np.newaxis]
            self.specgram = np.concatenate((self.specgram, np.abs(fft)), axis=1)

    def init_chirp(self):
        x = librosa.core.chirp(100, 1000, sr=22050, duration=5)
        x = x.astype('float32')
        self.chirp = x.tostring()

    def put_audio_data(self):
        self.output_stream.write(self.chirp)

    def plot(self):
        plt.imshow(self.specgram)
        plt.show()

    def start(self):
        self.init_chirp()
        self.iTh = th.Thread(target=self.get_audio_data)
        self.iTh2 = th.Thread(target=self.put_audio_data)
        self.iTh.start()
        self.iTh2.start()
        pass

    def stop(self):
        self.get_data = False


if __name__ == '__main__':
    a = Audiooo()
    # a.get_audio_data()

    a.start()
    time.sleep(5)
    a.stop()
    a.plot()
