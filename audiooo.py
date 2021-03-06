import time
import threading as th
import pyaudio as pa
import librosa
import matplotlib.pyplot as plt

import numpy as np


class Audiooo:

    def __init__(self, sr=22050):
        self.sr = sr
        self.buff_size = 1024
        self.buff = None

        self.specgram_h = self.buff_size // 2 + 1
        self.specgram_w = 1000
        self.specgram = np.zeros((self.specgram_h, self.specgram_w))

        self.frame_counter = 0


        self.pa = pa.PyAudio()
        self.input_stream = pa.Stream(PA_manager=self.pa, input=True, rate=self.sr, channels=1, format=pa.paInt16)
        self.output_stream = pa.Stream(PA_manager=self.pa, output=True, rate=self.sr, channels=1, format=pa.paFloat32)

        self.chirp = None
        self.get_data = True

    def get_audio_data(self):
        """
                Continuously get audio data from microphone and process data on the fly.
                The data will be stored in fft specgram form.

                stop by set self.get_data = False
                :return:
                """

        dt = np.dtype('int16')
        dt = dt.newbyteorder('>')

        while self.get_data is True:
            # read data
            _buff = self.input_stream.read(self.buff_size)
            self.buff = np.frombuffer(_buff, dtype=dt).astype('float32').copy() / 32767

            # TODO : window function &  overlap

            # process data
            fft = np.fft.fft(self.buff)[0:self.buff_size // 2 + 1]
            self.specgram[:, self.frame_counter] = abs(fft) ** 2
            self.frame_counter += 1

    def init_chirp(self):
        """
                Generate chirp voice by librosa.
                :return:
                """
        x = librosa.core.chirp(100, 1000, sr=22050, duration=5)
        x = x.astype('float32')
        self.chirp = x.tostring()

    def put_audio_data(self):
        """
                Play chirp voice.
                :return:
                """
        self.output_stream.write(self.chirp)

    def plot(self):

        plt.ion()
        # create axes
        ax1 = plt.subplot(111)

        # create image plot
        im1 = ax1.imshow(self.specgram)

        plt.ion()

        while self.get_data is True:
            im1.set_data(self.specgram)
            plt.pause(0.05)

        plt.ioff()
        plt.show()

    def start(self):
        """
                start everything in multi-thread mode.
                :return:
                """
        self.init_chirp()
        self.iTh = th.Thread(target=self.get_audio_data)
        self.iTh2 = th.Thread(target=self.put_audio_data)
        self.iTh3 = th.Thread(target=self.plot)
        self.iTh.start()
        self.iTh2.start()
        self.iTh3.start()
        pass

    def stop(self):
        """
                Stop recording.
                :return:
                """
        self.get_data = False


if __name__ == '__main__':
    a = Audiooo()
    # a.get_audio_data()
    a.start()
    time.sleep(20)
    a.stop()
