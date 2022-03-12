import sys
from pathlib import Path
import math
import pyqtgraph as pg
from pyqtgraph import PlotItem, GraphicsLayoutWidget

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QDoubleValidator
from live_streaming_ui import Ui_MainWindow

from typing import Optional
import pylsl
import numpy as np
import torch


plot_duration = 15  # how many seconds of data to show
update_interval = 60  # ms between screen updates
pull_interval = 90  # ms between each pull operation
control_interval = 100  # ms between sends of each control signal
win_len_sec = 0.750  # Length of processing window (750 ms)
blanking_ms = 500  # Length of waiting period after change in stim


class DataInletPlaceHolder:
    inlet = None

    def __init__(self):
        pass

    def set_inlet(self, info: pylsl.StreamInfo, layout: pg.GraphicsLayoutWidget):
        self.inlet = DataInlet(info, layout)

    def pull_and_plot(self, plot_time):
        if self.inlet is None:
            return
        self.inlet.pull_and_plot(plot_time)


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


class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""

    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]
    main_buffer = np.array([])

    def __init__(self, info: pylsl.StreamInfo, plot_layout: pg.GraphicsLayoutWidget):
        super().__init__(info)
        # calculate the size for our buffer, i.e. two times the displayed data
        bufsize = (
            2 * math.ceil(info.nominal_srate() * plot_duration),
            info.channel_count(),
        )
        # print(info.nominal_srate())
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])

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
        #     #self.curves.append(curve)

        # self.plots.append(win.addPlot())
        # win.nextRow()
        # self.curves.append(pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True))
        # self.plots[-1].addItem(self.curves[-1])
        # self.plots[-1].enableAutoRange(x=False, y=True)

    def pull_and_plot(self, plot_time):
        # pull the data
        _, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=self.buffer.shape[0], dest_obj=self.buffer
        )
        # ts will be empty if no samples were pulled, a list of timestamps otherwise
        if ts:
            ts = np.asarray(ts)
            y = self.buffer[0 : ts.size, :]
            this_x = None
            old_offset = 0
            new_offset = 0
            for ch_ix in range(self.channel_count):
                # we don't pull an entire screen's worth of data, so we have to
                # trim the old data and append the new data to it
                old_x, old_y = self.plots[ch_ix].curves[0].getData()
                # the timestamps are identical for all channels, so we need to do
                # this calculation only once
                if ch_ix == 0:
                    # find the index of the first sample that's still visible,
                    # i.e. newer than the left border of the plot
                    old_offset = old_x.searchsorted(plot_time)
                    # same for the new data, in case we pulled more data than
                    # can be shown at once
                    new_offset = ts.searchsorted(plot_time)
                    # append new timestamps to the trimmed old timestamps
                    this_x = np.hstack((old_x[old_offset:], ts[new_offset:]))
                # append new data to the trimmed old data

                y_new = y[new_offset:, ch_ix]

                # Save data copy:
                # _, old_y_data = self.curves_data[ch_ix].getData()
                # y_data = np.hstack([old_y_data[old_offset:],y_new.copy()])

                # y_new = (y_new - y_new.mean()) / y_new.std() - 5 * ch_ix
                this_y = np.hstack((old_y[old_offset:], y_new))
                # this_y = ( this_y - this_y.mean() ) / this_y.std()
                # print(this_y.shape)
                # replace the old data

                self.plots[ch_ix].curves[0].setData(this_x, this_y)
                # self.plots[ch_ix].curves[0].setPen((0, 0, 0, 0))

                try:
                    pass
                    # self.main_buffer[ch_ix] = this_y[-self.main_buffer.shape[-1] :].copy()
                    # print(self.main_buffer[:,-10:])
                except ValueError:
                    return


class ProcessorAndOutlet:
    model_off = None
    model_on = None
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

        self.score_curve_on = pg.PlotCurveItem(
            x=empty, y=empty, autoDownsample=True, pen=(60, 60, 250)
        )
        self.out_curve_on = pg.PlotCurveItem(
            x=empty, y=empty, autoDownsample=True, pen=(250, 0, 0)
        )
        plt_on.addItem(self.score_curve_on)
        plt_on.addItem(self.out_curve_on)

    def set_tresh_on(self, thresh):
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

                # If blanking, don't send anything
                return

        win_len = int(self.inlet.inlet.fs * win_len_sec)

        try:
            to_process = np.array(
                [plot.curves[0].getData()[1] for plot in self.inlet.inlet.plots]
            )
            if to_process.shape[-1] < win_len:
                return
            to_process = to_process[:, -win_len:]
            # to_process = filtfilt(b,a,data).copy()
        except (ValueError, RuntimeError):
            return

        # Select what model to use:
        if self.stim_is_on:
            model = self.model_on
        else:
            model = self.model_off

        x = torch.Tensor(to_process).unsqueeze(0)
        out_l = model(x).squeeze()
        out = torch.nn.functional.softmax(out_l, dim=0).squeeze()[1].item()
        # print(out)

        # Send to stream:
        last_out = self.out_cat
        if self.stim_is_on:
            self.out_cat = float(out > self.thresh_on)
        else:
            self.out_cat = float(out > self.thresh_off)
        self.outlet.push_sample([self.out_cat, self.out_cat])
        if last_out != self.out_cat and not finished_blanking:
            self.blanking = True

        # Update plots:
        if self.stim_is_on:
            score_curve, out_curve = self.score_curve_on, self.out_curve_on
        else:
            score_curve, out_curve = self.score_curve_off, self.out_curve_off
        # Score Curves:
        t_out, y_score = score_curve.getData()
        old_offset = t_out.searchsorted(plot_time)
        t_out, y_score = (
            np.hstack((t_out[old_offset:], np.array(pylsl.local_clock()))),
            np.hstack((y_score[old_offset:], np.array(out))),
        )
        score_curve.setData(t_out, y_score)

        # Out Curve:
        _, y_out_cat = out_curve.getData()
        y_out_cat = np.hstack((y_out_cat[old_offset:], np.array((self.out_cat))))
        out_curve.setData(
            t_out, y_out_cat,
        )
        out_curve.setPen((250, 0, 0))
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

        self.updateStreamsBtn.clicked.connect(self.updateStreams)
        self.StartStreamBtn.clicked.connect(self.startStreaming)

    def startStreaming(self):
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
        self_idx = self.stream_names.index(self.outlet.info.name())
        self.streams.remove(self.streams[self_idx])
        self.stream_names.remove(self.outlet.info.name())

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

    def getModelOFF(self):

        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "PT Files (*.pt)"
        )
        self.fname = Path(self.fname)
        model = torch.load(self.fname)
        model.eval()
        self.outlet.model_off = model

    def getModelON(self):
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "PT Files (*.pt)"
        )
        self.fname = Path(self.fname)

        model = torch.load(self.fname)
        model.eval()
        self.outlet.model_on = model


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
        mintime = pylsl.local_clock() - plot_duration
        # call pull_and_plot for each inlet.
        # Special handling of inlet types (markers, continuous data) is done in
        # the different inlet classes.

        inlet.pull_and_plot(mintime)

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

    # create a timer that will pull and add new data occasionally
    control_timer = QtCore.QTimer()
    control_timer.timeout.connect(update_out)
    control_timer.start(control_interval)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
