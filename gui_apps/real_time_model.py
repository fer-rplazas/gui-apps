import os
from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path
from xml.dom import ValidationErr
from sklearn.semi_supervised import SelfTrainingClassifier
import yaml

import joblib
import numpy as np
import torch
import warnings

from lfp_analysis import BipolarContructor, SignalFeatureExtractor
from scipy.signal import sosfilt, butter, sosfilt_zi


class RealTimeModelABC(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> float:
        """Extract features, and classify data"""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset any state if the model is being skipped"""
        pass


class StateFulFilter:
    def __init__(self, fs, cutoff_freq: float = 1.0):
        self.sos = butter(4, cutoff_freq, "highpass", fs=fs, output="sos")
        self.zi = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[np.newaxis, ...].transpose((1, 0, 2))
            self.zi = np.tile(self.zi, (1, x.shape[0], 1))

        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y

    def reset_state(self):
        self.zi = None


class RealTimePreProcessor:
    def __init__(
        self,
        bipolar: Optional[BipolarContructor] = None,
        filter: Optional[StateFulFilter] = None,
    ):

        self.bipolar_constructor = None
        self.filter = None

    def forward(self, x: np.ndarray) -> np.ndarray:

        if self.bipolar_constructor is not None:
            x = self.bipolar_constructor.form_bipolar(x)
        if self.filter is not None:
            x = self.filter.forward(x)

        return x


class RealTimeSVM(RealTimeModelABC):
    def __init__(self, fname: Union[str, os.PathLike], n_samp: int):
        s = joblib.load(fname)

        self.scaler = s["scaler"]
        self.feat_extractor: SignalFeatureExtractor = s["feat_extractor"]
        self.feat_extractor.prepare_realtime(n_samp)
        self.cls = s["cls"]
        self.feat_selection = s["feat_selection"]

    def forward(self, x: np.ndarray):

        x = self.feat_extractor.realtime_feats(x)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self.scaler.transform(x)
            x = x[self.feat_selection]

        return self.cls.predict(x)

    def reset_state(self) -> None:
        pass


class RealTimeCNN(RealTimeModelABC):
    def __init__(self, fname: Union[str, os.PathLike], n_samp, feat_selector, stats):

        self.model = torch.load(fname)
        self.model = self.model.eval()
        self.feat_selector = feat_selector
        self.feat_extractor = (
            SignalFeatureExtractor().prepare_realtime(n_samp).realtime_feats
        )
        self.means, self.stds = stats

        self.state = []

    def forward(self, x):

        feats = self.feat_extractor(x)
        feats = (feats - self.means) / self.stds
        feats = feats[self.feat_selector]

        with torch.no_grad():
            x = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
            feats = torch.Tensor(feats).unsqueeze(0)

            if not self.state:
                out_l, (h, c) = self.model.forward_realtime((x, feats))
                out_l.squeeze_()
            else:
                out_l, (h, c) = self.model.forward_realtime(
                    (x, feats), tuple(self.state)
                )
                out_l.squeeze_()
            self.state = list((h, c))
            return torch.sigmoid(out_l).squeeze().item()

    def reset_state(self) -> None:
        self.state = []


def load_model(
    fname: Union[str, os.PathLike], win_len_sec
) -> tuple[RealTimePreProcessor, Union[RealTimeSVM, RealTimeCNN], float]:

    report = joblib.load(Path(fname).with_suffix(".pkl"))

    bipolar = None
    if report["ch_names"] != report["ch_names_orig"]:
        bipolar = BipolarContructor(report["ch_names_orig"])

    preprocessor = RealTimePreProcessor(bipolar, StateFulFilter(report["fs"]))

    if Path(fname).suffix == ".svm":
        win_len_sec = report["svm_win"]
        model = RealTimeSVM(fname, n_samp=int(win_len_sec * report["fs"]))

    elif Path(fname).suffix == ".cnn":

        win_len_sec = report["cnn_win"]
        means, stds = report["data_mod/means"], report["data_mod/stds"]
        selected_feats = report["data_mod/feats"]

        model = RealTimeCNN(
            fname,
            n_samp=int(win_len_sec * report["fs"]),
            feat_selector=selected_feats,
            stats=(means, stds),
        )

    else:
        raise ValueError("Model extension not recognized")

    return preprocessor, model, win_len_sec
