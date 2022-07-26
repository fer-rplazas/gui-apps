import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pyqtgraph as pg
from lfp_analysis.data import BipolarContructor, SmrImporter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QFileDialog,
                             QFrame, QGridLayout, QGroupBox, QLabel, QLineEdit,
                             QMainWindow, QMenu, QVBoxLayout, QWidget)
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, iirnotch, lfilter, sosfilt, sosfilt_zi

from real_time_model import RealTimePreProcessor, StateFulFilter
from send_to_train import send_data_to_jade, train_and_return

TREMOR_BAND: bool = False


class ProcessingOptions:
    highpass = False
    notch50 = False
    notch100 = False

    def __init__(self) -> None:
        pass


class ChannelData:
    fs: float
    name: str
    data: np.ndarray
    plt: pg.PlotItem

    # Processing Options:
    processing_options_done: ProcessingOptions = ProcessingOptions()

    def __init__(self, name, data, plt, fs=2048.0, t=None) -> None:
        self.name = name
        self.data = data
        self.plt = plt
        plt.setLabel("left", name)
        self.fs = fs
        if t is None:
            self.t = np.linspace(0, (data.size - 1.0) / self.fs, data.size)
        else:
            self.t = t

        self.update_plot()

    def highpass(self, freq=1.0, N=6):
        if self.data.size > 0:
            self.data_backup = self.data.copy()
            sos = butter(N, freq, "highpass", fs=self.fs, output="sos")
            zi = sosfilt_zi(sos)
            self.data,_ = sosfilt(sos, self.data, zi=zi)

    def notch50(self, Q=30):
        if self.data.size > 0:
            self.data_backup = self.data.copy()
            b, a = iirnotch(50, Q, fs=self.fs)
            self.data = lfilter(b, a, self.data)

    def notch100(self, Q=30):
        if self.data.size > 0:
            self.data_backup = self.data.copy()
            b, a = iirnotch(100, Q, fs=self.fs)
            self.data = lfilter(b, a, self.data)

    def update(self, processing_options):
        update_plot = False
        if processing_options.highpass and not self.processing_options_done.highpass:
            self.highpass()
            update_plot = True

        if (not processing_options.highpass) and self.processing_options_done.highpass:
            self.data = self.data_backup
            update_plot = True

        if processing_options.notch50 and not self.processing_options_done.notch50:
            self.notch50()
            update_plot = True

        if (not processing_options.notch50) and self.processing_options_done.notch50:
            self.data = self.data_backup
            update_plot = True

        if processing_options.notch100 and not self.processing_options_done.notch100:
            self.notch100()
            update_plot = True

        if (not processing_options.notch100) and self.processing_options_done.notch100:
            self.data = self.data_backup
            update_plot = True

        if update_plot:
            self.update_plot()
        self.processing_options_done = processing_options

    def update_plot(self):
        # self.plt.plot(self.t, self.data)
        self.plt.curves[0].setData(self.t, self.data)
        self.plt.curves[0].setPen((150, 150, 150))
        pass


class ChannelGroup:
    channels: List[ChannelData]
    plt_area: Optional[pg.GraphicsLayoutWidget] = None

    # Processing Options:
    processing_options_done: ProcessingOptions = ProcessingOptions()

    def __init__(self, channels) -> None:
        self.channels = channels

    def reset(self):
        for plt in self.get_plots():
            plt.close()
            plt.parentItem().layout.removeItem(plt)
        pass

    def check_highpass(self, highpass_cmd, **kwargs):
        if highpass_cmd:
            for ch in self.channels:
                if not ch.highpassed:
                    ch.highpass(**kwargs, fs=ch.fs)

            self.highpassed = True

    def check_notched50(self, notch50_cmd, **kwargs):
        if notch50_cmd:
            for ch in self.channels:
                if not ch.notched50:
                    ch.notch50()

            self.notched50 = True

    def check_notched100(self, notch100_cmd, **kwargs):
        if notch100_cmd:
            for ch in self.channels:
                if not ch.notched100:
                    ch.notch100()

            self.notched100 = True

    def sort_against(self, sort_list: List[str]):
        names_old = self.ch_names().copy()
        self.channels.sort(key=lambda x: sort_list.index(x.name))

    def ch_names(self):
        if self.channels:
            return [ch.name for ch in self.channels]
        return []

    def get_plots(self):
        return [item.plt for item in self.channels]

    def __getitem__(self, name):
        try:
            idx = self.ch_names().index(name)
        except:
            return None

        item = self.channels[idx]
        return item

    def update_chans(self, ch_list, importer, ch_order, renew=False):
        if ch_list is None:
            return

        if renew or (isinstance(ch_list, list) and len(ch_list) == 0):
            self.reset()
        # if not ch_list:
        #     return

        # Check for eliminated chans:
        for my_ch in self.ch_names():
            if my_ch not in ch_list:
                item = self[my_ch]
                self.plt_area.removeItem(item.plt)
                self.channels.remove(item)

        for potential_ch in ch_list:
            if potential_ch not in self.ch_names() or renew:
                # Create Plot Item:
                plt = self.plt_area.addPlot(row=ch_order.index(potential_ch), col=0)
                plt.showAxis("bottom", False)
                plt.addItem(
                    pg.PlotCurveItem(
                        x=np.array([]), y=np.array([]), autoDownsample=True
                    )
                )
                # self.plt_area.nextRow()
                self.channels.append(
                    ChannelData(
                        potential_ch,
                        importer[potential_ch],
                        plt,
                        importer.fs,
                        importer.t(),
                    )
                )
                self.channels[-1].update(self.processing_options_done)
                self.sort_against(ch_order)
        self.update(self.processing_options_done)

    def update(self, processing_options):
        [ch.update(processing_options) for ch in self.channels]
        self.processing_options_done = processing_options

    def get_dataMat(self):
        return np.stack([item.data for item in self.channels])

    def reset(self):
        for ch in self.ch_names():
            item = self[ch]
            if item is not None:
                self.plt_area.removeItem(item.plt)
                self.channels.remove(item)


def update_all_regions(region):
    for plt in (
        data_model.lfps.get_plots()
        + data_model.peripherals.get_plots()
        + [data_model.peripherals.label_plot]
    ):
        plt.setXRange(*region.getRegion(), padding=0)


class Peripherals(ChannelGroup):
    label = np.ndarray([])
    label_score = np.ndarray([])
    smoothing_ms: float = 600
    thresh: float = 50

    def __init__(self, channels) -> None:
        super().__init__(channels)

    def reset(self):
        for ch in self.ch_names():
            item = self[ch]
            if item is not None:
                self.plt_area.removeItem(item.plt)
                self.channels.remove(item)

        if hasattr(self, "lr"):
            self.label_plot.close()
            self.label_plot.parentItem().layout.removeItem(self.label_plot)

            self.range_plot.close()
            self.range_plot.parentItem().layout.removeItem(self.range_plot)
        super().reset()

    def set_smoothing(self, smoothing_ms):
        if self.smoothing_ms == smoothing_ms:
            return
        self.smoothing_ms = smoothing_ms
        self.update(self.processing_options_done)

    def set_thresh(self, thresh):
        if self.thresh == thresh:
            return
        self.thresh = thresh
        self.update(self.processing_options_done)

    def get_label(self):
        if not self.channels:
            return

        data = np.stack([ch.data for ch in self.channels])
        norm = np.linalg.norm(data, axis=0)

        if TREMOR_BAND:
            sos = butter(4, [3, 7], "bandpass", output="sos", fs=self.channels[0].fs)
            norm = sosfilt(sos, norm)

        smoothed = uniform_filter1d(
            np.abs(norm), size=int(self.channels[0].fs * self.smoothing_ms / 1000)
        )
        thresh = np.percentile(smoothed, self.thresh)

        self.label_score = np.abs(smoothed)

        self.label = (
            uniform_filter1d(
                self.label_score > thresh,
                size=int(3 * self.channels[0].fs * self.smoothing_ms / 1000),
            )
            > 0.5
        )

    def set_rangeArea(self, rangeArea):
        self.rangeArea = rangeArea

    def update_chans(self, ch_list, importer, ch_order, renew=False):

        if ch_list is None:
            return

        if renew or (isinstance(ch_list, list) and len(ch_list) == 0):
            self.reset()

        if not hasattr(self, "label_plot"):
            self.label_plot = self.plt_area.addPlot(row=100, col=0)
            self.label_plot.setLabel("left", "Label")
            self.score_curve = pg.PlotCurveItem(autoDownsample=True)
            self.label_curve = pg.PlotCurveItem(autoDownsample=True)
            self.label_plot.addItem(self.score_curve)
            self.label_plot.addItem(self.label_curve)

        super().update_chans(ch_list, importer, ch_order)
        if not hasattr(self, "lr") and self.channels:
            self.range_plot = self.rangeArea.addPlot()
            self.range_plot.setLabel("left", "Range")

            # Range Plot underlying curve:
            self.range_plot_curve = pg.PlotCurveItem(autoDownsample=True)
            self.range_plot_curve.setData(self.channels[0].t, self.channels[0].data)
            self.range_plot_curve.setPen((150, 150, 150))
            self.range_plot.addItem(self.range_plot_curve)

            # Range Plot Selector Overlay:
            self.lr = pg.LinearRegionItem(
                [self.channels[0].t[0], self.channels[0].t[-1]]
            )
            self.lr.setBounds((self.channels[0].t[0], self.channels[0].t[-1]))
            self.lr.setZValue(-10)
            self.range_plot.addItem(self.lr)
            self.lr.sigRegionChanged.connect(lambda: update_all_regions(self.lr))

    def update(self, processing_options):
        super().update(processing_options)
        self.get_label()
        self.update_label_plot()

    def update_label_plot(self):
        if not self.channels:
            return
        self.score_curve.setData(self.channels[0].t, self.label_score)
        self.score_curve.setPen((100, 100, 240))
        self.label_curve.setData(
            self.channels[0].t, self.label * self.label_score.std() * 3
        )
        self.label_curve.setPen((250, 0, 0))
        if hasattr(self, "range_plot_curve"):
            self.range_plot_curve.setData(self.channels[0].t, self.label_score)


class DataModel:
    selected_lfps: List[str] = []
    selected_peri: List[str] = []

    importer: Optional[SmrImporter] = None
    lfp_candidates: List[str] = []
    peri_candidates: List[str] = []

    lfps: ChannelGroup = ChannelGroup([])
    peripherals: Peripherals = Peripherals([])
    t: np.ndarray = np.array([])
    # lfp_raw: LFPData = LFPData(np.array([]), names=[])
    # peripheral_raw: AccData = AccData(np.array([]), names=[])
    # t_high = np.array([])

    # lfp_processed = deepcopy(lfp_raw)
    # peripheral_processed = deepcopy(peripheral_raw)

    # lfp_plot = deepcopy(lfp_processed)
    # peripheral_plot = deepcopy(peripheral_processed)
    # t_plot = np.array([])

    # highpass, highpassed = False, False
    # notch50, notched50 = False, False
    # notch100, notched100 = False, False

    # label, label_score = np.array([]), np.array([])
    smoothing_ms, thresh = 200, 0.25

    def __init__(self) -> None:
        pass

    def reset(self):
        self.importer = None
        self.selected_lfps = []
        self.lfp_candidates = []
        self.peri_candidates = []
        self.selected_peri = []
        self.t = np.array([])
        self.lfps.reset()

        self.peripherals.reset()
        self.lfps = ChannelGroup([])
        self.peripherals = Peripherals([])
        # self.__init__()
        # self.setSelected([], [])
        # self.force_update()
        # TODO
        pass

    def force_update(self):
        self.lfps.update(ProcessingOptions())
        self.peripherals.update(ProcessingOptions())

    def setSelected(self, selected_lfps, selected_peri):

        self.selected_lfps, self.selected_peri = selected_lfps, selected_peri

        self.lfps.update_chans(
            selected_lfps, self.importer, self.lfp_candidates, renew=True
        )
        self.peripherals.update_chans(
            selected_peri, self.importer, self.peri_candidates, renew=True
        )

        # # self.lfp_raw = LFPData.from_importer(self.importer, self.selected_lfps)
        # # self.peripheral_raw = AccData.from_importer(self.importer, self.selected_peri)

        # # self.lfp_processed = deepcopy(self.lfp_raw)
        # # self.peripheral_processed = deepcopy(self.peripheral_raw)

        # # self.lfp_processed = deepcopy(self.lfp_raw)
        # # self.peripheral_processed = deepcopy(self.peripheral_raw)
        # # self.highpassed, self.notched50, self.notched100 = False, False, False

        # # self.processSigs()

    # def get_label(self):

    # if (
    #     not self.peripheral_processed.data.size > 0
    #     or self.smoothing_ms is None
    #     or self.thresh is None
    # ):
    #     return

    # norm = np.linalg.norm(self.peripheral_processed.data, axis=0)

    # smoothed = uniform_filter1d(
    #     np.abs(norm), size=int(self.importer.fs * self.smoothing_ms / 1000)
    # )
    # thresh = np.percentile(smoothed, self.thresh)

    # self.label_score = np.abs(smoothed)

    # self.label = (
    #     uniform_filter1d(
    #         self.label_score > thresh,
    #         size=int(3 * self.importer.fs * self.smoothing_ms / 1000),
    #     )
    #     > 0.5
    # )

    # def processSigs(self):

    # if self.highpass and not self.highpassed:
    #     self.lfp_processed.highpass()
    #     self.peripheral_processed.highpass()
    #     self.highpassed = True
    # elif not self.highpass and self.highpassed:
    #     self.setSelected(self.selected_lfps, self.selected_peri)

    # if self.notch50 and not self.notched50:
    #     self.lfp_processed.notch()
    #     self.peripheral_processed.notch()
    #     self.notched50 = True
    # elif not self.notch50 and self.notched50:
    #     self.setSelected(self.selected_lfps, self.selected_peri)

    # if self.notch100 and not self.notched100:
    #     self.lfp_processed.notch(100.0)
    #     self.peripheral_processed.notch(100.0)
    #     self.notched100 = True
    # elif not self.notch100 and self.notched100:
    #     self.setSelected(self.selected_lfps, self.selected_peri)

    # self.preparePlotSigs()
    # self.get_label()

    # def preparePlotSigs(self):
    # self.lfp_plot = deepcopy(self.lfp_processed)
    # # if not self.lfp_plot.decimated:
    # #     self.lfp_plot.decimate(10)

    # self.peripheral_plot = deepcopy(self.peripheral_processed)
    # # if not self.peripheral_plot.decimated:
    # #     self.peripheral_plot.decimate(10)

    def setImporter(self, importer):

        if self.importer is None:
            self.importer = importer
            ch_list = self.importer.ch_names
            self.lfp_candidates = [ch for ch in ch_list if ch[0] in ["L", "R"]]
            self.peri_candidates = [ch for ch in ch_list if ch[0] in ["A", "E"]]
            self.t = importer.t()
        else:
            self.importer = importer
            ch_list = self.importer.ch_names
            lfp_candidates = [ch for ch in ch_list if ch[0] in ["L", "R"]]
            peri_candidates = [ch for ch in ch_list if ch[0] in ["A", "E"]]
            self.t = importer.t()
            if (
                lfp_candidates == self.lfp_candidates
                and peri_candidates == self.peri_candidates
            ):
                # Trigger data refresh
                self.setSelected(self.selected_lfps, self.selected_peri)
            else:
                self.lfp_candidates = lfp_candidates
                self.peri_candidates = peri_candidates
                self.setSelected(self.selected_lfps, self.selected_peri)
                return True


# Initialize data model object:
data_model = DataModel()


class ChannelSelector(QWidget):
    layout = QVBoxLayout()
    lfp_chans = []
    peri_chans = []

    def __init__(self):
        super().__init__()
        self.setLayout(self.layout)

        # LFP Channel Selector:
        self.LFPChannelSelector = QGroupBox("LFP Channels:")
        self.LFPChannelSelectorLayout = QVBoxLayout()
        self.LFPChannelSelectorLayout.setAlignment(Qt.AlignTop)
        self.LFPChannelSelector.setLayout(self.LFPChannelSelectorLayout)

        # Peripheral Channel Selector:
        self.PeripheralChannelSelector = QGroupBox("Peripheral Channels:")
        self.PeripheralChannelSelectorLayout = QVBoxLayout()
        self.PeripheralChannelSelectorLayout.setAlignment(Qt.AlignTop)
        self.PeripheralChannelSelector.setLayout(self.PeripheralChannelSelectorLayout)

        # Put layout together:
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setLineWidth(0)

        self.layout.addWidget(self.LFPChannelSelector)
        self.layout.addWidget(separator)
        self.layout.addWidget(self.PeripheralChannelSelector)

    def add_lfp_chans(self, ch_names: List[str]):
        self.reset()
        for ch in ch_names:
            selector = QCheckBox(ch)
            selector.stateChanged.connect(self.selectionChanged)
            self.LFPChannelSelectorLayout.addWidget(selector)
            self.lfp_chans.append(selector)

    def add_peri_chans(self, ch_names: List[str]):
        for ch in ch_names:
            selector = QCheckBox(ch)
            selector.stateChanged.connect(self.selectionChanged)
            self.PeripheralChannelSelectorLayout.addWidget(selector)
            self.peri_chans.append(selector)

    def selectionChanged(self):
        selected_lfps, selected_peri = [], []
        for sel in self.lfp_chans + self.peri_chans:
            if sel.isChecked():
                if sel.text() in data_model.lfp_candidates:
                    selected_lfps.append(sel.text())
                elif sel.text() in data_model.peri_candidates:
                    selected_peri.append(sel.text())
                else:
                    raise ValueError("Unknown Channel selected")

        if not selected_lfps:
            selected_lfps = None
        if not selected_peri:
            selected_peri = None

        data_model.lfps.update_chans(
            selected_lfps,
            data_model.importer,
            data_model.lfp_candidates,
        )
        data_model.peripherals.update_chans(
            selected_peri, data_model.importer, data_model.peri_candidates
        )

    def reset(self):
        for selector in self.lfp_chans + self.peri_chans:
            selector.close()

        self.lfp_chans, self.peri_chans = [], []


class LFPProcessingArea(QWidget):
    layout = QVBoxLayout()

    def __init__(self):
        super().__init__()

        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

        self.highPassCheck = QCheckBox("Highpass")
        self.highPassCheck.stateChanged.connect(self.processingChanged)
        self.notch50Check = QCheckBox("Notch 50")
        self.notch50Check.stateChanged.connect(self.processingChanged)
        self.notch100Check = QCheckBox("Notch 100")
        self.notch100Check.stateChanged.connect(self.processingChanged)
        self.bipolarCheck = QCheckBox("Bipolar")
        self.bipolarCheck.stateChanged.connect(self.processingChanged)

        # Set layout:
        self.layout.addWidget(self.highPassCheck)
        self.layout.addWidget(self.notch50Check)
        self.layout.addWidget(self.notch100Check)
        self.layout.addWidget(self.bipolarCheck)

    def processingChanged(self):
        processing_state = ProcessingOptions()
        if self.highPassCheck.isChecked():
            processing_state.highpass = True

        if self.notch50Check.isChecked():
            processing_state.notch50 = True

        if self.notch100Check.isChecked():
            processing_state.notch100 = True

        if self.bipolarCheck.isChecked():
            data_model.do_bipolar = True

        data_model.lfps.update(processing_state)


class PeriProcessingArea(QWidget):
    layout = QVBoxLayout()

    def __init__(self):
        super().__init__()

        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

        self.highPassCheck = QCheckBox("Highpass")
        self.highPassCheck.stateChanged.connect(self.processingChanged)

        self.tremorBandcheck = QCheckBox("3-7 Hz")
        self.tremorBandcheck.stateChanged.connect(self.processingChangedTremor)

        self.smoothingLabel = QLabel("Smoothing [ms]:")
        self.smoothingInput = QLineEdit()
        self.smoothingInput.setValidator(QDoubleValidator())
        self.smoothingInput.setText(str(600.0))
        self.smoothing_val = float(self.smoothingInput.text())
        self.smoothingChanged()
        self.smoothingInput.editingFinished.connect(self.smoothingChanged)

        self.threshLabel = QLabel("Threshold (percentile):")
        self.threshInput = QLineEdit()
        self.threshInput.setValidator(QDoubleValidator())
        self.threshInput.setText(str(50))
        self.thresh_val = float(self.threshInput.text())
        self.threshChanged()
        self.threshInput.editingFinished.connect(self.threshChanged)

        # Set layout:
        self.layout.addWidget(self.highPassCheck)
        self.layout.addWidget(self.tremorBandcheck)
        self.layout.addWidget(self.smoothingLabel)
        self.layout.addWidget(self.smoothingInput)
        self.layout.addWidget(self.threshLabel)
        self.layout.addWidget(self.threshInput)

    def processingChangedTremor(self):
        global TREMOR_BAND
        TREMOR_BAND = self.tremorBandcheck.isChecked()

    def processingChanged(self):
        processing_state = ProcessingOptions()
        if self.highPassCheck.isChecked():
            processing_state.highpass = True

        data_model.peripherals.update(processing_state)

    def smoothingChanged(self):
        new_val = float(self.smoothingInput.text())
        if self.smoothing_val == new_val:
            return

        data_model.peripherals.set_smoothing(new_val)
        self.smoothing_val = new_val

    def threshChanged(self):
        new_val = float(self.threshInput.text())
        if self.thresh_val == new_val:
            return

        data_model.peripherals.set_thresh(new_val)
        self.thresh_val = new_val


class SignalsArea(QWidget):
    layout = QGridLayout()

    def __init__(self):
        super().__init__()
        self.setLayout(self.layout)

        self.RangeArea = pg.GraphicsLayoutWidget(show=True)
        self.RangeArea.setBackground("w")

        self.LFPArea = pg.GraphicsLayoutWidget(show=True)
        self.LFPArea.setBackground("w")
        self.LFPProcessingArea = LFPProcessingArea()

        self.PeriArea = pg.GraphicsLayoutWidget(show=True)
        self.PeriArea.setBackground("w")
        self.PeriProcessingArea = PeriProcessingArea()

        # First Column:
        self.layout.addWidget(self.RangeArea, 0, 0, 10, 1)
        self.layout.addWidget(self.LFPArea, 10, 0, 1, 1)
        self.layout.addWidget(self.PeriArea, 12, 0, 2, 1)
        # Second Column:
        self.layout.addWidget(self.LFPProcessingArea, 10, 1, 1, 5)
        self.layout.addWidget(self.PeriProcessingArea, 12, 1, 2, 5)

        # Add to data_model
        data_model.lfps.plt_area = self.LFPArea
        data_model.peripherals.plt_area = self.PeriArea
        data_model.peripherals.set_rangeArea(self.RangeArea)

    def reset(self):
        # Add to data_model
        data_model.lfps.plt_area = self.LFPArea
        data_model.peripherals.plt_area = self.PeriArea
        data_model.peripherals.set_rangeArea(self.RangeArea)


class Window(QMainWindow):
    """Main Window."""

    data = DataModel()

    def __init__(self):
        """Initializer."""
        super().__init__()
        self.setWindowTitle("Label Extraction")
        self.resize(1270, 720)

        # Prepare layout:
        self.layout = QGridLayout()
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Channel Selector:
        self.channelSelector = ChannelSelector()
        self.layout.addWidget(self.channelSelector, 0, 0)

        # Signals and Processing Options Area:
        self.SignalsArea = SignalsArea()
        self.layout.addWidget(self.SignalsArea, 0, 1)

        # self.init_range = True  # Flag indicating that

        # self.SignalsAreaLayout.addWidget(self.RangeArea, 0, 0, 10, 1)
        # self.SignalsAreaLayout.addWidget(self.SignalArea, 10, 0, 1, 1)
        # self.SignalsAreaLayout.addWidget(self.PeriArea, 12, 0, 2, 1)
        # # self.SignalsAreaLayout.addWidget(self.LabelArea, 14, 0, 7, 1)

        # # Processing Options:
        # self.LFPProcessingArea = QWidget()
        # self.LFPProcessingAreaLayout = QVBoxLayout()
        # self.LFPProcessingAreaLayout.setAlignment(Qt.AlignTop)
        # self.LFPProcessingArea.setLayout(self.LFPProcessingAreaLayout)

        # self.highPassCheck = QCheckBox("Highpass")
        # self.highPassCheck.stateChanged.connect(self.highPassChanged)
        # self.notch50Check = QCheckBox("Notch 50")
        # self.notch50Check.stateChanged.connect(self.notch50Changed)
        # self.notch100Check = QCheckBox("Notch 100")
        # self.notch100Check.stateChanged.connect(self.notch100Changed)

        # self.LFPProcessingAreaLayout.addWidget(self.highPassCheck)
        # self.LFPProcessingAreaLayout.addWidget(self.notch50Check)
        # self.LFPProcessingAreaLayout.addWidget(self.notch100Check)

        # self.SignalsAreaLayout.addWidget(self.LFPProcessingArea, 10, 1, 1, 1)

        # # Label Extraction Options:
        # self.LabelProcessingArea = QWidget()
        # self.LabelProcessingAreaLayout = QVBoxLayout()
        # self.LabelProcessingAreaLayout.setAlignment(Qt.AlignTop)
        # self.LabelProcessingArea.setLayout(self.LabelProcessingAreaLayout)

        # self.smoothingSlider = QLineEdit()
        # self.smoothingSlider.setMaximumWidth(48)
        # self.smoothingSlider.setValidator(QDoubleValidator())
        # self.smoothingSlider.setText(str(500.0))
        # # self.smoothingSlider.setMinimum(100)
        # self.smoothingSlider.editingFinished.connect(self.smoothingChanged)
        # self.LabelProcessingAreaLayout.addWidget(self.smoothingSlider)

        # self.threshSlider = QLineEdit()
        # self.threshSlider.setMaximumWidth(48)
        # self.threshSlider.setValidator(QDoubleValidator())
        # self.threshSlider.setText(str(50))
        # # self.threshSlider.setRange(1,100)s
        # self.threshSlider.editingFinished.connect(self.threshChanged)
        # self.LabelProcessingAreaLayout.addWidget(self.threshSlider)

        # self.SignalsAreaLayout.addWidget(self.LabelProcessingArea, 12, 1, 2, 5)

        self._createActions()
        self._createMenuBar()

    # def smoothingChanged(self):

    #     self.data.smoothing_ms = float(self.smoothingSlider.text())  # 50 - 5000 ms
    #     self.data.get_label()
    #     self.plotPeri()

    # def threshChanged(self):

    #     self.data.thresh = float(self.threshSlider.text())  # ) * 0.03 + 0.07 # 0.1 -- 3
    #     self.data.get_label()
    #     self.plotPeri()

    # def highPassChanged(self):

    #     if self.highPassCheck.isChecked():
    #         self.data.highpass = True
    #         self.data.processSigs()
    #     else:
    #         self.data.highpass = False
    #         self.data.processSigs()

    #     self.plotLFPs()
    #     self.plotPeri()

    # def notch50Changed(self):

    #     if self.notch50Check.isChecked():
    #         self.data.notch50 = True
    #         self.data.processSigs()
    #     else:
    #         self.data.notch50 = False
    #         self.data.processSigs()

    #     self.plotLFPs()
    #     self.plotPeri()

    # def notch100Changed(self):

    # #     if self.notch50Check.isChecked():
    # #         self.data.notch100 = True
    # #         self.data.processSigs()
    # #     else:
    # #         self.data.notch100 = False
    # #         self.data.processSigs()

    # #     self.plotLFPs()
    # #     self.plotPeri()

    # # def updateXRanges(self):

    #     for plot in self.peri_plots + self.lfp_plots:
    #         plot.setXRange(*self.linearRegion.getRegion(), padding=0)

    # def plotLFPs(self):
    #     self.SignalArea.clear()

    #     self.lfp_plots = []
    #     for ch in self.data.lfp_plot.names:
    #         self.lfp_plots.append(
    #             self.SignalArea.addPlot(
    #                 y=self.data.lfp_plot.data[self.data.lfp_plot.names.index(ch)],
    #                 autoDownsample=True,
    #             )
    #         )
    #         self.lfp_plots[-1].setLabel("left", ch)
    #         self.lfp_plots[-1].showAxis("bottom", False)
    #         self.SignalArea.nextRow()

    # def plotPeri(self):

    #     # self.LabelArea.clear()
    #     # self.labelPlot = self.LabelArea.addPlot(y=self.data.label.astype(float))
    #     # self.labelPlot.plot(self.data.label_score)
    #     # self.labelPlot.plot(self.data.label)

    #     self.PeriArea.clear()
    #     self.peri_plots = []
    #     for ch in self.data.peripheral_plot.names:
    #         self.peri_plots.append(
    #             self.PeriArea.addPlot(
    #                 y=self.data.peripheral_plot.data[
    #                     self.data.peripheral_plot.names.index(ch)
    #                 ],
    #                 autoDownsample=True,
    #             )
    #         )
    #         if not self.init_range:
    #             self.linearRegion.sigRegionChanged.connect(self.updateXRanges)
    #         self.peri_plots[-1].setLabel("left", ch)
    #         self.peri_plots[-1].showAxis("bottom", False)
    #         self.PeriArea.nextRow()

    #     self.peri_plots.append(
    #         self.PeriArea.addPlot(y=self.data.label_score, autoDownsample=True)
    #     )
    #     if self.data.label.size > 0:
    #         self.peri_plots[-1].plot(
    #             self.data.label.astype(float) * 2 * self.data.label_score.std()
    #             + self.data.label_score.mean(),
    #             autoDownsample=True,
    #             pen=(255, 0, 0),
    #         )
    #     self.peri_plots[-1].setLabel("left", "Label")
    #     self.peri_plots[-1].showAxis("bottom", False)

    #     if self.init_range and self.data.peripheral_plot.data.size > 0:
    #         self.RangePlot = self.RangeArea.addPlot(y=self.data.peripheral_plot.data[0])
    #         self.linearRegion = pg.LinearRegionItem(
    #             [0, len(self.data.peripheral_plot.data[0])],
    #             bounds=(0, len(self.data.peripheral_plot.data[0])),
    #         )
    #         self.RangePlot.showAxis("left", False)
    #         self.linearRegion.setZValue(-10)
    #         self.RangePlot.addItem(self.linearRegion)

    #         self.init_range = not self.init_range
    #         self.plotPeri()
    #         self.plotLFPs()

    def fileOpen(self):

        self.fname, _ = QFileDialog.getOpenFileName(
            self, "Please select an SMR file", "", "SMR Files (*.smr)"
        )
        self.fname = Path(self.fname)

        if data_model.importer is None:
            data_model.setImporter(SmrImporter(self.fname))
            lfp_candidates = data_model.lfp_candidates
            peri_candidates = data_model.peri_candidates

            self.channelSelector.add_lfp_chans(lfp_candidates)
            self.channelSelector.add_peri_chans(peri_candidates)

        else:
            retVal = data_model.setImporter(SmrImporter(self.fname))
            if retVal is not None:
                lfp_candidates = data_model.lfp_candidates
                peri_candidates = data_model.peri_candidates
                self.channelSelector.add_lfp_chans(lfp_candidates)
                self.channelSelector.add_peri_chans(peri_candidates)

    #         self.selector_list = []
    #         for ch in lfp_candidates:
    #             selector = QCheckBox(ch)
    #             # selector.stateChanged.connect(lambda x: self.checkedState)
    #             self.selector_list.append(selector)
    #             self.LFPChannelSelectorLayout.addWidget(selector)
    #             self.selector_list[-1].stateChanged.connect(self.checkedState)

    #         for ch in peri_candidates:
    #             selector = QCheckBox(ch)
    #             self.selector_list.append(selector)
    #             self.PeripheralChannelSelectorLayout.addWidget(selector)
    #             self.selector_list[-1].stateChanged.connect(self.checkedState)

    #     def checkedState(self):

    #         selected_lfps, selected_peri = [], []
    #         for sel in self.selector_list:
    #             if sel.isChecked():
    #                 if sel.text() in self.data.lfp_candidates:
    #                     selected_lfps.append(sel.text())

    #                 elif sel.text() in self.data.peri_candidates:
    #                     selected_peri.append(sel.text())
    #                 else:
    #                     raise RuntimeError(
    #                         "Invalid Channel encountered during checkbox callback"
    #                     )

    #         # Call data Model:
    #         self.data.setSelected(selected_lfps, selected_peri)

    #         self.plotLFPs()
    #         self.plotPeri()

    def exportData(self, choose_file=True):

        if choose_file:
            fname, _ = QFileDialog.getSaveFileName(
                self, "QFileDialog.getSaveFileName()", "", "H5 File (*.h5)"
            )

            self.fname = Path(fname)

        start_t, end_t = data_model.peripherals.lr.getRegion()
        start_idx = data_model.peripherals.channels[0].t.searchsorted(start_t)
        end_idx = data_model.peripherals.channels[0].t.searchsorted(end_t)

        f = h5py.File(self.fname.with_suffix(".h5"), "w")

        lfps = data_model.lfps.get_dataMat()[:, start_idx:end_idx]
        ch_names_orig = deepcopy(data_model.lfps.ch_names())
        ch_names = data_model.lfps.ch_names()

        if hasattr(data_model, "do_bipolar"):
            if data_model.do_bipolar:
                bip = BipolarContructor(data_model.lfps.ch_names())
                lfps = bip.form_bipolar(lfps)
                ch_names = bip.out_names

        f.create_dataset("LFP", data=lfps)
        f.create_dataset("ch_names", data=ch_names)
        f.create_dataset("ch_names_orig", data=ch_names_orig)
        f.create_dataset("label", data=data_model.peripherals.label[start_idx:end_idx])
        f.create_dataset("fs", data=np.array(data_model.lfps.channels[0].fs))
        f.close()

    def train(self):

        self.exportData(choose_file=False)
        send_data_to_jade(f'''"{str(self.fname.with_suffix(".h5"))}"''')
        train_and_return(f'''"{self.fname.with_suffix(".h5")}"''')

    def _createActions(self):
        self.openAction = QAction("&Open File...", self)
        self.openAction.triggered.connect(self.fileOpen)

        self.exportAction = QAction("&Export to H5...", self)
        self.exportAction.triggered.connect(self.exportData)

        self.trainAction = QAction("&Train on jade...", self)
        self.trainAction.triggered.connect(self.train)

        self.closeAction = QAction("Close File...")
        self.closeAction.triggered.connect(self.reset)

    def _createMenuBar(self):
        menuBar = self.menuBar()

        # Creating menus using a QMenu object
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.exportAction)
        fileMenu.addAction(self.trainAction)
        fileMenu.addAction(self.closeAction)

    def reset(self):
        # self.SignalsArea.reset()
        self.channelSelector.reset()

        data_model.reset()

        self.SignalsArea.reset()
        data_model.peripherals.set_rangeArea(self.SignalsArea.RangeArea)


data = DataModel()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
