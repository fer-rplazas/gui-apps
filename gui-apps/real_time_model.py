import os
from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path
from xml.dom import ValidationErr
import yaml

import joblib
import numpy as np
import torch

from lfp_analysis import BipolarContructor
from scipy.signal import sosfilt, butter, sosfilt_zi


class RealTimeModelABC(ABC):

    @abstractmethod
    def load(self, fname: Union[str, os.PathLike]) -> None:
        """Load Model from file"""
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> float:
        """Extract features, and classify data"""
        pass

    def reset_state(self) -> None:
        """Reset any state if the model is being skipped"""
        pass


class StateFulFilter:
    def __init__(self, fs, cutoff_freq: float = 1.0):
        self.sos = butter(6, cutoff_freq, "highpass", fs=fs, output="sos")
        self.zi = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)

        y,self.zi= sosfilt(self.sos, x, self.zi)
        return y

    def reset_state(self):
        self.zi = None


class RealTimePreProcessor:
    def __init__(
        self,
        bipolar: Optional[BipolarContructor] = None,
        filter: Optional[StateFulFilter] = None
    ):

        self.bipolar_constructor = bipolar
        self.filter = filter

    def forward(self, x: np.ndarray) -> np.ndarray:

        if self.bipolar_constructor is not None:
            x = self.bipolar_constructor.form_bipolar(x)
        if self.filter is not None:
            x = self.filter.forward(x)


class RealTimeSVM(RealTimeModelABC):
    def __init__(self, fname: Union[str, os.PathLike]):
        pass

    def reset_state(self) -> None:
        pass

class RealTimeCNN(RealTimeModelABC):
    
    def __init__(self, fname: Union[str, os.PathLike]):
        
        self.model = torch.load(fname)
        self.model = self.model.eval()

        self.state = []

    def forward(self, x):

        with torch.no_grad():
            x = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
            
            if not self.state:
                out_l,(h,c) = self.model.forward_realtime(x)
                out_l.squeeze_()
            else:
                out_l,(h,c) = self.model.forward_realtime(x,tuple(self.state))
                out_l.squeeze_()
            self.state = list((h,c))
            return torch.sigmoid(out_l).squeeze().item()

    def reset_state(self) -> None:
        self.state = []


def load_model(fname: Union[str, os.PathLike]) -> tuple[RealTimePreProcessor, Union[RealTimeSVM, RealTimeCNN]]:

    global win_len_sec
    with open(Path(fname).with_suffix('.yaml'),'r') as f:
        report = yaml.safe_load(f)

    bipolar = None
    if report['ch_names'] != report['ch_names_orig']:
        bipolar = BipolarContructor(report['ch_names_orig'])
    
    preprocessor = RealTimePreProcessor(bipolar, StateFulFilter(report['fs']))
    
    if Path(fname).suffix == '.svm':
        model = RealTimeSVM(fname)
        
        win_len_sec = report['svm_win']

    elif Path(fname).suffix == '.cnn':
        model = RealTimeCNN(fname)
        
        win_len_sec = report['cnn_win']
    else:
        raise ValueError('Model extension not recognized')

    return preprocessor, model
    