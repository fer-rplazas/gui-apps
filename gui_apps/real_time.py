import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pylsl
import pyqtgraph as pg
import torch
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtGui import QDoubleValidator
from scipy import signal
from copy import deepcopy
import warnings

np.seterr(all='warn')
import logging

logging.basicConfig(level=logging.INFO, filename="output.log")

from real_time_model import *

Ui_MainWindow, _ = uic.loadUiType(Path("ui_files") / "live_streaming_ui.ui")
_ = torch.load(
    "saved_models/dummy_model_13_chan.pt"
)  # Load dummy model before showing UI (first torch.load call is slow)

plot_duration = 10  # how many seconds of data to show
update_interval = 50  # ms between screen updates
pull_interval = 90  # ms between each pull operation
plot_interval = int(1 / 24.0 * 1000) # 24 Hz plotting 
control_interval = 250  # ms between sends of each control signal
win_len_sec = 0.75 # Length of processing window (750 ms)
blanking_ms = 0.0  # ms of waiting period after change in stim


class DataInletPlaceHolder:
    inlet = None
    HP_filter, notch50, notch100, notch150 = False, False, False, False

    def __init__(self):
        pass

    def set_inlet(self, info: pylsl.StreamInfo, layout: pg.GraphicsLayoutWidget):
        self.inlet = DataInlet(info, layout)
        self.inlet.HP_filter = self.HP_filter

    def set_HP_filter(self, HP_filter: bool):
        self.HP_filter = HP_filter
        if self.inlet is not None:
            self.inlet.HP_filter = HP_filter

    def set_notch(self, do50, do100, do150):

        self.notch50, self.notch100, self.notch150 = do50, do100, do150
        if self.inlet is not None:
            self.inlet.notch50, self.inlet.notch100, self.inlet.notch150 = (
                do50,
                do100,
                do150,
            )

    def plot(self):
        if self.inlet is None:
            return
        self.inlet.plot()

    def pull(self):
        if self.inlet is None:
            return
        self.inlet.pull()


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
            max_buflen=plot_duration,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.fs = info.nominal_srate()

    def pull_and_plot(self, plot_time: float, plt: pg.PlotItem):
        """Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        pass


class SignalBuffer:
    def __init__(self, n_chan: int, max_samps: int):
        self.X = np.empty((n_chan, 0))
        self.t = np.empty((0))
        self.max_samps = max_samps

    def push(self, ts, x):
        self.X = np.append(self.X, x, axis=-1)
        self.t = np.append(self.t, ts, axis=-1)

        if self.X.shape[-1] > self.max_samps or self.t.size > self.max_samps:
            self.X = self.X[:, int(self.X.shape[-1] - self.max_samps) :]
            self.t = self.t[int(self.t.shape[-1] - self.max_samps) :]


class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""

    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]
    HP_filter, notch50, notch100, notch150 = False, False, False, False

    def __init__(self, info: pylsl.StreamInfo, plot_layout: pg.GraphicsLayoutWidget):
        super().__init__(info)
        # calculate the size for our buffer, i.e. two times the displayed data
        bufsize = (
            2 * math.ceil(info.nominal_srate() * plot_duration),
            info.channel_count(),
        )
        # print(info.nominal_srate())
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])

        self.signal_buffer = SignalBuffer(info.channel_count(), self.fs * 20)
        # Create plots and curves:
        empty = np.array([])

        # self.curves = [pg.PlotCurveItem(x=empty, y=empty, autoDownSample=True) for _ in range(self.channel_count)]
        # for curve in self.curves:
        #     plt.addItem(curve)

        # pass
        self.plots = []
        for jj in range(info.channel_count()):
            plot = plot_layout.addPlot(title=None)
            plot.enableAutoRange(x=False, y=True)
            plot.showAxis("bottom", False)
            curve = pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True)
            plot.addItem(curve)
            plot_layout.nextRow()

            self.plots.append(plot)

        # HP Filter Coeffs:
        self.sos = signal.butter(8, 0.5, "highpass", output="sos", fs=self.fs)
        self.filter_coeff = [
            signal.sosfilt_zi(self.sos) for _ in range(info.channel_count())
        ]

        self.notch50params = signal.iirnotch(50, 30, fs=self.fs)
        self.notch50coeff = [
            signal.lfilter_zi(*self.notch50params) for _ in range(info.channel_count())
        ]

        self.notch100params = signal.iirnotch(100, 30, fs=self.fs)
        self.notch100coeff = [
            signal.lfilter_zi(*self.notch100params) for _ in range(info.channel_count())
        ]

        self.notch150params = signal.iirnotch(150, 30, fs=self.fs)
        self.notch150coeff = [
            signal.lfilter_zi(*self.notch150params) for _ in range(info.channel_count())
        ]

        self.ts = np.array([0])

        self.counter = 0
        #     #self.curves.append(curve)

        # self.plots.append(win.addPlot())
        # win.nextRow()
        # self.curves.append(pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True))
        # self.plots[-1].addItem(self.curves[-1])
        # self.plots[-1].enableAutoRange(x=False, y=True)

    def pull(self):
        samps, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=4096 * 30, dest_obj=None
        )

        if ts:
            self.signal_buffer.push(np.asarray(ts), np.asarray(samps).T)

    def plot(self):
        # pull the data
        # _, ts = self.inlet.pull_chunk(
        #     timeout=0.0, max_samples=self.buffer.shape[0], dest_obj=self.buffer
        # )

        # ts will be empty if no samples were pulled, a list of timestamps otherwise

        for ch_ix in range(self.channel_count):
            self.plots[ch_ix].curves[0].setData(
                self.signal_buffer.t[-int(plot_duration * self.fs) :],
                self.signal_buffer.X[ch_ix, -int(plot_duration * self.fs) :],
            )

        # if ts:

        #     self.signal_buffer.push(np.asarray(ts), np.asarray(samps).T)

        #     # print(self.signal_buffer.X.shape)
        #     # old_ts = deepcopy(self.ts)
        #     ts = np.asarray(ts)
        #     # print(ts.shape)

        #     # print(ts[0] in old_ts)

        #     # self.ts = ts
        #     # y = self.buffer[0 : ts.size, :]
        #     y = np.asarray(samps)
        #     this_x = None
        #     old_offset = 0
        #     new_offset = 0
        #     for ch_ix in range(self.channel_count):
        #         # we don't pull an entire screen's worth of data, so we have to
        #         # trim the old data and append the new data to it
        #         old_x, old_y = self.plots[ch_ix].curves[0].getData()
        #         # the timestamps are identical for all channels, so we need to do
        #         # this calculation only once
        #         if ch_ix == 0:
        #             # find the index of the first sample that's still visible,
        #             # i.e. newer than the left border of the plot
        #             old_offset = old_x.searchsorted(plot_time)
        #             # same for the new data, in case we pulled more data than
        #             # can be shown at once
        #             new_offset = ts.searchsorted(plot_time)
        #             # append new timestamps to the trimmed old timestamps
        #             this_x = np.hstack((old_x[old_offset:], ts[new_offset:]))
        #         # append new data to the trimmed old data

        #         y_new = y[new_offset:, ch_ix]

        #         if self.HP_filter:
        #             y_new, self.filter_coeff[ch_ix] = signal.sosfilt(
        #                 self.sos, y_new, zi=self.filter_coeff[ch_ix]
        #             )

        #         if self.notch50:
        #             y_new, self.notch50coeff[ch_ix] = signal.lfilter(
        #                 *self.notch50params, y_new, zi=self.notch50coeff[ch_ix]
        #             )

        #         if self.notch100:
        #             y_new, self.notch100coeff[ch_ix] = signal.lfilter(
        #                 *self.notch100params, y_new, zi=self.notch100coeff[ch_ix]
        #             )

        #         if self.notch150:
        #             y_new, self.notch150coeff[ch_ix] = signal.lfilter(
        #                 *self.notch150params, y_new, zi=self.notch150coeff[ch_ix]
        #             )
        #         # Save data copy:
        #         # _, old_y_data = self.curves_data[ch_ix].getData()
        #         # y_data = np.hstack([old_y_data[old_offset:],y_new.copy()])

        #         # y_new = (y_new - y_new.mean()) / y_new.std() - 5 * ch_ix
        #         this_y = np.hstack((old_y[old_offset:], y_new))
        #         # this_y = ( this_y - this_y.mean() ) / this_y.std()
        #         # print(this_y.shape)
        #         # replace the old data

        #         self.plots[ch_ix].curves[0].setData(this_x, this_y)
        #         # self.plots[ch_ix].curves[0].setPen((0, 0, 0, 0))


class ProcessorAndOutlet:
    model_off = None
    model_on = None
    preprocessor = None
    stim_is_on = False
    blanking = False
    blanking_counter: int = 0
    blank_for: int = int(math.ceil(float(blanking_ms) / control_interval))
    thresh_off: float = 0.5
    thresh_on: float = 0.5
    out_cat = 0

    def __init__(
        self,
        inlet,
        plt_off: pg.PlotItem,
        plt_on: pg.PlotItem,
        name="cnn_control_signal",
    ):

        self.inlet = inlet
        self.info = pylsl.StreamInfo(name, "", 2, 1.0 / (control_interval / 1000))

        channels = self.info.desc().append_child("channels")
        for c in ["L", "R"]:
            channels.append_child("channel").append_child_value("label", c)

        self.outlet = pylsl.StreamOutlet(self.info, chunk_size=1)

        empty = np.array([])
        # OFF Plots:
        self.score_curve_off = pg.PlotCurveItem(
            x=empty, y=empty, autoDownsample=True, pen=(60, 60, 250)
        )
        self.out_curve_off = pg.PlotCurveItem(
            x=empty, y=empty, autoDownsample=True, pen=(250, 0, 0)
        )
        plt_off.addItem(self.score_curve_off)
        plt_off.addItem(self.out_curve_off)
        plt_off.enableAutoRange(x=False, y=False)
        plt_off.setYRange(0, 1, padding=0.05)

        self.score_curve_on = pg.PlotCurveItem(
            x=empty, y=empty, autoDownsample=True, pen=(60, 60, 250)
        )
        self.out_curve_on = pg.PlotCurveItem(
            x=empty, y=empty, autoDownsample=True, pen=(250, 0, 0)
        )
        plt_on.addItem(self.score_curve_on)
        plt_on.addItem(self.out_curve_on)
        plt_on.enableAutoRange(x=False, y=False)
        plt_on.setYRange(0, 1, padding=0.05)

        self.model_on_state = []
        self.model_off_state = []

    def blankingChanged(self, new_t: float):
        self.blank_for = int(math.ceil(float(new_t) / control_interval))

    def set_thresh_on(self, thresh):
        self.thresh_on = thresh

    def set_thresh_off(self, thresh):
        self.thresh_off = thresh

    def process_and_send(self, plot_time):
        if self.model_off is None or self.model_on is None or self.inlet.inlet is None:
            return

        finished_blanking = False
        if self.blanking:
            self.blanking_counter += 1
            if self.blanking_counter > self.blank_for:
                self.blanking = False
                self.blanking_counter = 0
                finished_blanking = True
                pass
            else:
                return  # If blanking, don't send anything

        win_len = int(self.inlet.inlet.fs * win_len_sec)

        try:
            to_process = np.array(
                [plot.curves[0].getData()[1] for plot in self.inlet.inlet.plots]
            )
            if to_process.shape[-1] < win_len:
                return
            to_process = to_process[:, -win_len:]

            logging.info(to_process.shape)

            # to_process = filtfilt(b,a,data).copy()
        except (ValueError, RuntimeError):
            return

        # Preprocess data:
        if self.preprocessor is not None:
            to_process = self.preprocessor.forward(to_process)

        # Select what model to use:
        if self.stim_is_on:
            model = self.model_on
            other_model = self.model_off
        else:
            model = self.model_off
            other_model = self.model_on

        out = model.forward(to_process)
        other_model.reset_state()


        # Send to stream:
        last_out = self.out_cat
        if self.stim_is_on:
            self.out_cat = float(out > self.thresh_on)
        else:
            self.out_cat = float(out > self.thresh_off)
        self.outlet.push_sample([self.out_cat, self.out_cat])
        if last_out != self.out_cat and not finished_blanking:
            self.blanking = True

        logging.info(f"{pylsl.local_clock()}; {self.out_cat}")

        # Update plots:
        if self.stim_is_on:
            score_curve, out_curve = self.score_curve_on, self.out_curve_on
            other_score, other_out = self.score_curve_off, self.out_curve_off
        else:
            score_curve, out_curve = self.score_curve_off, self.out_curve_off
            other_score, other_out = self.score_curve_on, self.out_curve_on
        # Score Curves:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t_out, y_score = score_curve.getData()
            old_offset = t_out.searchsorted(plot_time)
            t_out, y_score = (
                np.hstack((t_out[old_offset:], np.array(pylsl.local_clock()))),
                np.hstack((y_score[old_offset:], np.array(out))),
            )
            score_curve.setData(t_out, y_score, connect="finite")

            # Out Curve:
            _, y_out_cat = out_curve.getData()
            y_out_cat = np.hstack((y_out_cat[old_offset:], np.array((self.out_cat))))
            out_curve.setData(t_out, y_out_cat, connect="finite")
            out_curve.setPen((250, 0, 0))

            # Other Curves:
            _, other_y = other_score.getData()
            _, other_cat = other_out.getData()

            other_y = np.hstack((other_y[old_offset:], np.array(np.nan)))
            other_score.setData(t_out, other_y, connect="finite")

            other_cat = np.hstack((other_cat[old_offset:], np.array(np.nan)))
            other_out.setData(t_out, other_cat, connect="finite")
            other_out.setPen((250, 0, 0))

        self.stim_is_on = bool(self.out_cat)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setupUi(self)

        self.selectModelOFFBtn.clicked.connect(self.getModelOFF)
        self.selectModelONBtn.clicked.connect(self.getModelON)

        self.inputThreshOFF.editingFinished.connect(self.threshOffChanged)
        self.inputThreshOFF.setValidator(QDoubleValidator(0.0, 1.0, 100))
        self.inputThreshON.editingFinished.connect(self.threshOnChanged)
        self.inputThreshON.setValidator(QDoubleValidator(0.0, 1.0, 100))

        self.BlankingTimeInput.editingFinished.connect(self.blankingChanged)
        self.BlankingTimeInput.setValidator(QDoubleValidator(0.0, 5000, 500))

        self.updateStreamsBtn.clicked.connect(self.updateStreams)
        self.StartStreamBtn.clicked.connect(self.startStreaming)

        self.StopBtn.clicked.connect(self.stop)

        self.HPCheckBox.clicked.connect(self.hpChanged)
        self.notch50Btn.clicked.connect(self.notchChanged)
        self.notch100Btn.clicked.connect(self.notchChanged)
        self.notch150Btn.clicked.connect(self.notchChanged)

    def stop(self):
        if hasattr(self, "timers"):
            [timer.stop() for timer in self.timers]

    def blankingChanged(self):
        self.outlet.blankingChanged(float(self.BlankingTimeInput.text()))

    def hpChanged(self):
        self.inlet.set_HP_filter(self.HPCheckBox.isChecked())

    def notchChanged(self):
        self.inlet.set_notch(
            self.notch50Btn.isChecked(),
            self.notch100Btn.isChecked(),
            self.notch150Btn.isChecked(),
        )

    def startStreaming(self):

        if hasattr(self, "timers"):
            self.timers[0].start(update_interval)
            self.timers[1].start(pull_interval)
            self.timers[2].start(plot_interval)
            self.timers[3].start(control_interval)

        selected_stream = self.streamList.selectedItems()
        if not selected_stream:
            return
        selected_stream = selected_stream[0].text()
        if not selected_stream:
            return

        stream_idx = self.stream_names.index(selected_stream)
        self.inlet.set_inlet(self.streams[stream_idx], self.SignalsPlot)

    def updateStreams(self):
        self.streams = pylsl.resolve_streams()
        self.stream_names = [stream.name() for stream in self.streams]

        try:
            self_idx = self.stream_names.index(self.outlet.info.name())
            self.streams.remove(self.streams[self_idx])
            self.stream_names.remove(self.outlet.info.name())
        except ValueError:
            pass

        self.streamList.clear()
        self.streamList.addItems(self.stream_names)

    def threshOnChanged(self):
        self.outlet.set_thresh_on(float(self.inputThreshON.text()))

    def threshOffChanged(self):
        self.outlet.set_thresh_off(float(self.inputThreshOFF.text()))

    def set_inlet(self, inlet):
        self.inlet = inlet

    def set_outlet(self, outlet):
        self.outlet = outlet
        self.blankingChanged()

    def getModelOFF(self):
        global win_len_sec

        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", filter="*.cnn *.svm"
        )
        self.fname = Path(self.fname)
        self.outlet.preprocessor, self.outlet.model_off, win_len_sec = load_model(self.fname,win_len_sec)

    def getModelON(self):
        global win_len_sec
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", filter="*.cnn *.svm"
        )
        self.fname = Path(self.fname)
        self.outlet.preprocessor, self.outlet.model_on, win_len_sec = load_model(self.fname,win_len_sec)


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()

    inlet = DataInletPlaceHolder()
    outlet = ProcessorAndOutlet(inlet, main.modelOFFPlot, main.modelONPlot)

    main.set_inlet(inlet)
    main.set_outlet(outlet)

    main.show()

    def scroll():
        if inlet.inlet is None:
            return
        """Move the view so the data appears to scroll"""
        # We show data only up to a timepoint shortly before the current time
        # so new data doesn't suddenly appear in the middle of the plot
        fudge_factor = pull_interval * 0.002
        plot_time = pylsl.local_clock()
        [
            plt.setXRange(
                plot_time - plot_duration + fudge_factor, plot_time - fudge_factor
            )
            for plt in [*inlet.inlet.plots, main.modelOFFPlot, main.modelONPlot]
        ]

    def update_in():
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        #mintime = pylsl.local_clock() - plot_duration
        # call pull_and_plot for each inlet.
        # Special handling of inlet types (markers, continuous data) is done in
        # the different inlet classes.

        inlet.pull()

    def plot_in():
        inlet.plot()

    def update_out():
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        mintime = pylsl.local_clock() - plot_duration
        # call pull_and_plot for each inlet.
        # Special handling of inlet types (markers, continuous data) is done in
        # the different inlet classes.
        outlet.process_and_send(mintime)

    # create a timer that will move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    # create a timer that will pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update_in)
    pull_timer.start(pull_interval)

    plot_timer = QtCore.QTimer()
    plot_timer.timeout.connect(plot_in)
    plot_timer.start(plot_interval)

    # create a timer that will pull and add new data occasionally
    control_timer = QtCore.QTimer()
    control_timer.timeout.connect(update_out)
    control_timer.start(control_interval)

    main.timers = [update_timer, pull_timer, plot_timer, control_timer]

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
