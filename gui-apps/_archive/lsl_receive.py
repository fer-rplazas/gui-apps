import numpy as np
import math
import pylsl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from typing import List, Optional

import torch

from pathlib import Path
from scipy.signal import butter, filtfilt
from zmq import EMTHREAD

b, a = butter(3, 1, "highpass", fs=4096.0)

# NN Settings:
MODEL_PATH = Path("C:\\Users\\pblab\\ET_Test\\data.pt")

# Basic parameters for the plotting window
plot_duration = 15  # how many seconds of data to show
update_interval = 60  # ms between screen updates
pull_interval = 120  # ms between each pull operation
control_interval = 100  # ms between sends of each control signal

STREAM_IN_NAME = "Alek_LFP"


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

    def __init__(self, info: pylsl.StreamInfo, win: pg.PlotWindow):
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
            plot = win.addPlot(title=None)
            plot.enableAutoRange(x=False, y=True)
            curve = pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True)
            plot.addItem(curve)
            win.nextRow()

            self.plots.append(plot)
        #     #self.curves.append(curve)

        # self.plots.append(win.addPlot())
        # win.nextRow()
        # self.curves.append(pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True))
        # self.plots[-1].addItem(self.curves[-1])
        # self.plots[-1].enableAutoRange(x=False, y=True)

    def set_main_buffer(self, main_buffer):
        self.main_buffer = main_buffer

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
    def __init__(
        self,
        buffer: np.ndarray,
        models: List[torch.nn.Module],
        inlet,
        name="CNN_outputs",
        plt: Optional[pg.PlotItem] = None,
    ):

        self.buffer = buffer
        self.models = models
        self.inlet = inlet

        self.info = pylsl.StreamInfo(name, "", 2, 1.0 / (control_interval / 1000))

        channels = self.info.desc().append_child("channels")
        for c in ["L", "R"]:
            channels.append_child("channel").append_child_value("label", c)

        self.outlet = pylsl.StreamOutlet(self.info, chunk_size=1)

        empty = np.array([])
        self.out_curve = pg.PlotCurveItem(
            x=empty, y=empty, autoDownsample=True, pen=(250, 0, 0)
        )
        plt.addItem(self.out_curve)
        self.out_cat_curve = pg.PlotCurveItem(
            x=empty, y=empty, autoDownsample=True, pen=(250, 0, 0)
        )

        plt.addItem(self.out_cat_curve)

    def process_and_send(self, plot_time):

        try:
            to_process = np.array(
                [plot.curves[0].getData()[1] for plot in self.inlet.plots]
            )
            if to_process.shape[-1] < 1535:
                return
            to_process = to_process[:, -1535:]
            # to_process = filtfilt(b,a,data).copy()
        except (ValueError, RuntimeError):
            return

        x = torch.Tensor(to_process).unsqueeze(0)
        out_l = self.models[0](x).squeeze()
        out = torch.nn.functional.softmax(out_l, dim=0).squeeze()[1].item()
        # print(out)

        # Send to stream:
        out_cat = float(out > 0.5)
        self.outlet.push_sample([out_cat, out_cat])

        # Update plot:
        t_out, y_out = self.out_curve.getData()
        _, y_out_cat = self.out_cat_curve.getData()
        old_offset = t_out.searchsorted(plot_time)
        t_out, y_out = (
            np.hstack((t_out[old_offset:], np.array(pylsl.local_clock()))),
            np.hstack((y_out[old_offset:], np.array(out))),
        )
        y_out_cat = np.hstack((y_out_cat[old_offset:], np.array((out_cat))))
        self.out_curve.setData(
            t_out, y_out, pen=(200, 200, 200), symbolBrush=(255, 0, 0), symbolPen="w"
        )
        self.out_cat_curve.setData(
            t_out,
            y_out_cat,
            pen=(200, 200, 200),
            symbolBrush=(255, 0, 0),
            symbolPen="w",
        )
        self.out_cat_curve.setPen((250, 0, 0))


class MarkerInlet(Inlet):
    """A MarkerInlet shows events that happen sporadically as vertical lines"""

    def __init__(self, info: pylsl.StreamInfo):
        super().__init__(info)

    def pull_and_plot(self, plot_time, plt):
        # TODO: purge old markers
        strings, timestamps = self.inlet.pull_chunk(0)
        if timestamps:
            for string, ts in zip(strings, timestamps):
                plt.addItem(
                    pg.InfiniteLine(ts, angle=90, movable=False, label=string[0])
                )


def main():

    # Create the pyqtgraph window
    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True)
    win.resize(1000, 600)
    win.setWindowTitle("Real-time data stream")

    # plt_signals = win.addPlot(title="LFPs")
    # plt_signals.enableAutoRange(x=False, y=True)
    # win.nextRow()

    # Resolve Streams:
    streams = pylsl.resolve_streams()
    inlet = None
    for stream in streams:
        if stream.name() == STREAM_IN_NAME:
            inlet = DataInlet(stream, win)

    if inlet is None:
        raise RuntimeError("Stream not found")

    # Initialize plots:
    plt_control = win.addPlot(title="Classifier output")
    plt_control.enableAutoRange(x=False, y=True)

    main_buf_size = (inlet.channel_count, math.ceil(inlet.fs * plot_duration / 2))
    main_buffer = np.zeros(main_buf_size, dtype=float) * np.nan

    inlet.set_main_buffer(main_buffer)

    # Load Models and prepare control stream:
    model_on = torch.load(MODEL_PATH)
    model_on.eval()
    # model_off = torch.load(PATH_TO_MODEL_OFF)
    models = []
    models.append(model_on)

    outlet = ProcessorAndOutlet(main_buffer, models, inlet, plt=plt_control)

    # iterate over found streams, creating specialized inlet objects that will
    # handle plotting the data
    # for info in streams:
    #     if info.type() == 'Markers':
    #         if info.nominal_srate() != pylsl.IRREGULAR_RATE \
    #                 or info.channel_format() != pylsl.cf_string:
    #             print('Invalid marker stream ' + info.name())
    #         print('Adding marker inlet: ' + info.name())
    #         inlets.append(MarkerInlet(info))
    #     elif info.nominal_srate() != pylsl.IRREGULAR_RATE \
    #             and info.channel_format() != pylsl.cf_string:
    #         print('Adding data inlet: ' + info.name())
    #         inlets.append(DataInlet(info, plts[:-1]))
    #     else:
    #         print('Don\'t know what to do with stream ' + info.name())

    def scroll():
        """Move the view so the data appears to scroll"""
        # We show data only up to a timepoint shortly before the current time
        # so new data doesn't suddenly appear in the middle of the plot
        fudge_factor = pull_interval * 0.002
        plot_time = pylsl.local_clock()
        [
            plt.setXRange(
                plot_time - plot_duration + fudge_factor, plot_time - fudge_factor
            )
            for plt in [*inlet.plots, plt_control]
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

    import sys

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()


if __name__ == "__main__":
    main()
