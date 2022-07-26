import sys
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QDoubleValidator

from win_control import get_coords, clicks, clicks_alt

import pylsl
import numpy as np

import os
from pathlib import Path

clear = lambda: os.system("cls")
Ui_MainWindow, _ = uic.loadUiType(Path("ui_files") / "toggle_stim.ui")

pull_interval = 20  # Read from LSL stream every ms


class InletPlaceholder:
    inlet = None

    def __init__(self):
        pass

    def set_inlet(self, info: pylsl.StreamInfo):
        self.inlet = DataInlet(info)

    def pull_and_process(self):
        if self.inlet is None:
            return
        self.inlet.pull_and_process()


class Inlet:
    """Base class to represent a plottable inlet"""

    def __init__(self, info: pylsl.StreamInfo):
        # create an inlet and connect it to the outlet we found earlier.
        # max_buflen is set so data older the plot_duration is discarded
        # automatically and we only pull data new enough to show it

        # Also, perform online clock synchronization so all streams are in the
        # same time domain as the local lsl_clock()
        # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
        # and dejitter timestamps
        self.inlet = pylsl.StreamInlet(
            info,
            max_buflen=512,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.fs = info.nominal_srate()

    def pull_and_process(self):
        """Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        pass


class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted asmultiple lines."""

    last_y = False

    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo):
        super().__init__(info)
        # calculate the size for our buffer, i.e. two times the displayed data

    def pull_and_process(self):
        # pull the data
        y, ts = self.inlet.pull_chunk(timeout=0.0, max_samples=64)
        # ts will be empty if no samples were pulled, a list of timestamps otherwise
        if ts:
            y = bool(y[0][0])
            print(f"Last y: {self.last_y}")
            print(f"New y: {y}")
            print(f"Equiv: {self.last_y != y}")
            print()

            if self.last_y != y:
                clicks_alt()
                self.last_y = y


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    stim_is_on = False

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.TurnStimONBtn.clicked.connect(self.turnStimON)
        self.TurnStimOFFBtn.clicked.connect(self.turnStimOFF)
        self.UpdateCoordsBtn.clicked.connect(self.updateCoords)

        self.UpdateStreamsBtn.clicked.connect(self.updateStreams)

        self.StartBtn.clicked.connect(self.start)
        self.StopBtn.clicked.connect(self.stop)

    def updateStreams(self):

        self.streams = pylsl.resolve_streams()
        self.stream_names = [stream.name() for stream in self.streams]

        #         try:
        #             # self_idx = self.stream_names.index(self.outlet.info.name())
        #             # self.streams.remove(self.streams[self_idx])
        #             self.stream_names.remove(self.outlet.info.name())
        #         except ValueError:
        #             pass

        self.StreamList.clear()
        self.StreamList.addItems(self.stream_names)

    def stop(self):
        self.inlet.inlet = None
        pass

    def start(self):

        selected_stream = self.StreamList.selectedItems()
        if not selected_stream:
            return
        selected_stream = selected_stream[0].text()
        if not selected_stream:
            return

        stream_idx = self.stream_names.index(selected_stream)
        self.inlet.set_inlet(self.streams[stream_idx])

    def updateCoords(self):
        get_coords()

    def turnStimON(self):
        if not self.stim_is_on:
            clicks_alt()
            self.stim_is_on = True

    def turnStimOFF(self):
        if self.stim_is_on:
            clicks_alt()
            self.stim_is_on = False


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()

    inlet = InletPlaceholder()
    main.inlet = inlet

    main.show()

    def update_in():
        inlet.pull_and_process()

    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update_in)
    pull_timer.start(pull_interval)

    sys.exit(app.exec_())
