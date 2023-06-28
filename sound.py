import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
from PyQt5 import uic

UI_FILE = "sound_ai_ui3.ui"

class AudioPlayerApp(QMainWindow):
    def __init__(self):
        super(AudioPlayerApp, self).__init__()
        self.ui = uic.loadUi(UI_FILE, self)
        self.media_player = QMediaPlayer(self)
        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_slider_range)
        self.slider = self.ui.horizontalSlider
        self.slider.setTracking(True)
        self.open_button = self.ui.pushButton_5
        self.open_button.clicked.connect(self.open_audio)
        self.play_button = self.ui.pushButton
        self.play_button.clicked.connect(self.play_audio)
        self.stop_button = self.ui.pushButton_4
        self.stop_button.clicked.connect(self.stop_audio)
        self.slider.sliderMoved.connect(self.set_position)
        self.SAMPLE_RATE = 22050
        self.NUM_SAMPLES = 22050
        self.waveform_button = self.ui.pushButton_6
        self.spectrogram_button = self.ui.pushButton_9
        self.mfcc_button = self.ui.pushButton_7
        self.fourier_button = self.ui.pushButton_8
        self.stacked_layout = self.ui.graph_stack
        self.sub_graph = self.ui.stackedWidget
        self.waveform_button.clicked.connect(lambda: self.stacked_layout.setCurrentIndex(0))
        self.spectrogram_button.clicked.connect(lambda: self.stacked_layout.setCurrentIndex(1))
        self.mfcc_button.clicked.connect(lambda: self.stacked_layout.setCurrentIndex(2))
        self.fourier_button.clicked.connect(lambda: self.stacked_layout.setCurrentIndex(3))

    def open_audio(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Audio Files (*.mp3 *.wav)")
        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()[0]
            print(f'File Path {self.file_path}')
            self.media_player.setMedia(QMediaContent(QtCore.QUrl.fromLocalFile(self.file_path)))
            self.plot_waveform(self.file_path)
            self.slider.setValue(0)
            duration = self.media_player.duration()
            self.slider.setRange(0, duration)
            self.media_player.positionChanged.connect(self.update_slider_position)
            self.play_button.setEnabled(True)

    def clear_graphs(self):
        for i in reversed(range(self.stacked_layout.count())):
            widget = self.stacked_layout.widget(i)
            self.stacked_layout.removeWidget(widget)
            widget.deleteLater()

    def plot_waveform(self, audio_file):
        self.clear_graphs()
        y, sr = librosa.load(audio_file)
        plt.style.use('dark_background')
        waveform = y
        time = np.arange(0, len(waveform)) / sr
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram))
        fourier = np.fft.fft(y)
        waveform_figure = plt.figure()
        waveform_canvas = FigureCanvas(waveform_figure)
        waveform_figure.clear()
        waveform_ax = waveform_figure.add_subplot(111)
        waveform_ax.plot(time, waveform)
        waveform_ax.set_title('Waveform')
        waveform_ax.set_xlabel('Time (s)')
        waveform_ax.set_ylabel('Amplitude')
        waveform_canvas.draw()
        self.stacked_layout.addWidget(waveform_canvas)
        spectrogram_figure = plt.figure()
        spectrogram_canvas = FigureCanvas(spectrogram_figure)
        spectrogram_figure.clear()
        spectrogram_ax = spectrogram_figure.add_subplot(111)
        librosa.display.specshow(librosa.power_to_db(spectrogram), sr=sr, x_axis='time', y_axis='mel', ax=spectrogram_ax)
        spectrogram_ax.set_title('Spectrogram')
        spectrogram_canvas.draw()
        self.stacked_layout.addWidget(spectrogram_canvas)
        mfcc_figure = plt.figure()
        mfcc_canvas = FigureCanvas(mfcc_figure)
        mfcc_figure.clear()
        mfcc_ax = mfcc_figure.add_subplot(111)
        librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=mfcc_ax)
        mfcc_ax.set_title('MFCC')
        mfcc_canvas.draw()
        self.stacked_layout.addWidget(mfcc_canvas)
        fourier_figure = plt.figure()
        fourier_canvas = FigureCanvas(fourier_figure)
        fourier_figure.clear()
        fourier_ax = fourier_figure.add_subplot(111)
        fourier_ax.plot(np.abs(fourier))
        fourier_ax.set_title('Fourier Transform')
        fourier_canvas.draw()
        self.stacked_layout.addWidget(fourier_canvas)
        self.stacked_layout.setCurrentIndex(0)

    def play_audio(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText('Play')
        else:
            self.media_player.play()
            self.play_button.setText('Pause')

    def stop_audio(self):
        self.media_player.stop()
        self.play_button.setText('Play')

    def update_slider_range(self, duration):
        self.slider.setRange(0, duration)

    def update_slider_position(self, position):
        self.slider.setValue(position)

    def set_position(self, position):
        self.media_player.setPosition(position)
    
    def play_audio(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText('Play')
        else:
            self.media_player.play()
            self.play_button.setText('Pause')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    audio_player = AudioPlayerApp()
    audio_player.showMaximized()
    sys.exit(app.exec_())

