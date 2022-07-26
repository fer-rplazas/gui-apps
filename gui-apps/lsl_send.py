import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet



def from_array(data: np.ndarray, fs: float, chunk_size: int = 32):
    """Send array as real-time LSL stream in infinite loop

    Args:
        data (np.ndarray): LFPs of shape [n_chan x n_samples]
        fs (float): Sampling rate of the data
        chunk_size (int): number of samples to be sent out at once to stream (LSL sending frequency will be adjusted accordingly)
    """    
    name = "Data Stream"
    type_ = "LFPs"
    n_chan = data.shape[0]

    info = StreamInfo(name, type_, n_chan, fs, "float32", "myUID111")

    outlet = StreamOutlet(info, chunk_size=chunk_size)

    print("sending data now ...")

    while True:
        
        for n in range(data.shape[-1]//chunk_size):
            
            samples = data[:,int(n*chunk_size):int((n+1)*chunk_size)].tolist()

            outlet.push_chunk(samples)

            time.sleep(chunk_size / fs)


def random_lfps():
    """Generate uniformally distributed random samples and send them out as LSL data stream"""

    fs = 4096.0
    name = "Test Stream"
    type_ = "LFPs"
    n_chan = 14

    chunk_size = 32

    info = StreamInfo(name, type_, n_chan, fs, "float32", "myUID111")
    outlet = StreamOutlet(info, chunk_size=chunk_size)

    print("sending data now ...")
    while True:

        samples = [
            (np.random.rand(chunk_size) + 10_000).tolist() for _ in range(n_chan)
        ]

        outlet.push_chunk(samples)

        time.sleep(chunk_size / fs)


def random_stim():
    """Generate stimulation bursts of random duration and send them out as real-time LSL data stream"""

    fs = 50
    name = "random_stim"
    type_ = "cntrol signals"
    n_chan = 2

    info = StreamInfo(name, type_, n_chan, fs, "float32", "myUID00001")
    outlet = StreamOutlet(info)

    durations = np.random.uniform(0.5, 5.0, 300) # Define burst length properties here

    time_course = []
    offset = 0
    for dur in durations:
        time_course.extend(list(0.1 + offset + np.zeros((int(dur * fs)))))
        offset = 1 if offset==0 else 0

    print("sending data now...")
    while True:
        for el in time_course:
            outlet.push_sample([el, el])
            time.sleep(1.0 / fs)


if __name__ == "__main__":
    random_stim()
