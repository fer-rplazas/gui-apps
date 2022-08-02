# %%

from gui_apps.lsl_send import benchmark_model

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt



# %% [markdown]
# 

# %%
fname = """C:\\UsersData\\Fernando\\OneDrive - Nexus365\\Recordings and Data\\KS_0717_PD_CNN_aDBS\\KS_0717_Finger_to_nose_OFF_test.h5"""
fname = """C:\\UsersData\\Fernando\\OneDrive - Nexus365\\Recordings and Data\\KS_0717_PD_CNN_aDBS\\KS_0717_Finger_to_Nose_ON.cnn"""
fname_svm = """C:\\UsersData\\Fernando\\OneDrive - Nexus365\\Recordings and Data\\KS_0717_PD_CNN_aDBS\\KS_0717_Finger_to_Nose_ON.svm"""

# fname = """C:\\UsersData\\Fernando\\OneDrive - Nexus365\\Recordings and Data\\KS_0717_PD_CNN_aDBS\\KS_0717_Finger_to_Nose_ON_test.h5"""

fname = """C:\\UsersData\\Fernando\\OneDrive - Nexus365\\Recordings and Data\\KS_0717_PD_CNN_aDBS\\KS_0717_Finger_to_Nose_randStim.cnn"""
fname_svm = """C:\\UsersData\\Fernando\\OneDrive - Nexus365\\Recordings and Data\\KS_0717_PD_CNN_aDBS\\KS_0717_Finger_to_Nose_randStim.svm"""

# %%
# label: np.ndarray = h5py.File(fname, "r")["label"][:]
# lfps: np.ndarray = h5py.File(fname, "r")["LFP"][:]
# fs: float = h5py.File(fname, "r")["fs"][()]
# ch_names: list[str] = list(h5py.File(fname, "r")["ch_names"][:])
# ch_names_orig: list[str] = list(h5py.File(fname, "r")["ch_names_orig"][:])

# %%
# lfps.shape,fs

# %%
benchmark_model(fname, fname_svm)


