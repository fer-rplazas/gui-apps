# %%

from gui_apps.lsl_send import from_array

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt



# %% [markdown]
# 

# %%
fname = """C:\\UsersData\\Fernando\\OneDrive - Nexus365\\Recordings and Data\\KS_0717_PD_CNN_aDBS\\KS_0717_Finger_to_nose_OFF_test.h5"""
fname = """C:\\UsersData\\Fernando\\OneDrive - Nexus365\\Recordings and Data\\KS_0717_PD_CNN_aDBS\\KS_0717_Finger_to_Nose_ON.h5"""
# fname = """C:\\UsersData\\Fernando\\OneDrive - Nexus365\\Recordings and Data\\KS_0717_PD_CNN_aDBS\\KS_0717_Finger_to_Nose_ON_test.h5"""

fname = """C:\\UsersData\\Fernando\\OneDrive - Nexus365\\Recordings and Data\\KS_0717_PD_CNN_aDBS\\KS_0717_Finger_to_Nose_randStim.h5"""

# %%
label: np.ndarray = h5py.File(fname, "r")["label"][:]
lfps: np.ndarray = h5py.File(fname, "r")["LFP"][:]
fs: float = h5py.File(fname, "r")["fs"][()]
ch_names: list[str] = list(h5py.File(fname, "r")["ch_names"][:])
ch_names_orig: list[str] = list(h5py.File(fname, "r")["ch_names_orig"][:])

# %%
lfps.shape,fs

# %%
from_array(lfps,fs,y=label)


