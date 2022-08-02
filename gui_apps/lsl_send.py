import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams

from .real_time_model import *

import logging

def y_to_file():

    logging.basicConfig(filename='y_stream.log', level=logging.INFO, filemode='w+')

    fs = 4096
    info = StreamInfo('y_stream', '', 1, fs, "double64", "myUID112")

    inlet = StreamInlet(info,max_buflen=4096)


    fs_pull = int(fs/10)

    while True:

        samples, time_stamps = inlet.pull_chunk()

        logging.info(f"{time_stamps[-1]};{samples[-1]}")

        time.sleep(1.0 / fs_pull)


from real_time import SignalBuffer

def benchmark_model(fname_model_cnn, fname_model_svm):

    fileh = logging.FileHandler('stream_benchmark.log', 'w+')
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    logging.info("Timestamps;CNN;SVM;y_true")
    
    
    win_len_sec_cnn, win_len_sec_svm = 0.0, 0.0
    ctrl_interval = 0.250

    preprocessor, model_cnn, win_len_sec_cnn = load_model(fname_model_cnn, win_len_sec_cnn)

    preprocessor, model_svm, win_len_sec_svm = load_model(fname_model_svm, win_len_sec_svm)

    streams = resolve_streams()

    info_data = [stream for stream in streams if stream.name() == "Data Stream"][0]
    info_y = [stream for stream in streams if stream.name() == "y_stream"][0]

    n_chan = info_data.channel_count()
    

    buffer_data = SignalBuffer(n_chan, 10*info_data.nominal_srate())

    inlet_data = StreamInlet(info_data,max_buflen=4096)
    inlet_y = StreamInlet(info_y,max_buflen=4096)

    counter = 0
    while True:

        # print(counter)

        samps,ts = inlet_data.pull_chunk(0.0)

        ys, ts_ys = inlet_y.pull_chunk(0.0)

        if ts:
            buffer_data.push(np.asarray(ts),np.asarray(samps).T)

            if counter > 10:

                x = buffer_data.X[:,-int(win_len_sec_cnn * info_data.nominal_srate()):]
                x = preprocessor.forward(x)

                out = model_cnn.forward(x)


                x = buffer_data.X[:,-int(win_len_sec_svm * info_data.nominal_srate()):]
                x = preprocessor.forward(x)
                out_svm = model_svm.forward(x)

                # print(out)

                logging.info(f"{float(np.asarray(ts)[-1]):3f};{float(out):.2f};{float(out_svm):.2f};{float(np.asarray(ys)[-1]):.1f}")

        counter +=1
        time.sleep(ctrl_interval)




        








def from_array(data: np.ndarray, fs: float, chunk_size: int = 1024, y:np.ndarray = None):
    """Send array as real-time LSL stream in infinite loop

    Args:
        data (np.ndarray): LFPs of shape [n_chan x n_samples]
        y (np.ndarray): Related data to send out in another stream (e.g. labels)
        fs (float): Sampling rate of the data
        chunk_size (int): number of samples to be sent out at once to stream (LSL sending frequency will be adjusted accordingly)
    """
    name = "Data Stream"
    type_ = "LFPs"
    n_chan = data.shape[0]

    info = StreamInfo(name, type_, n_chan, fs, "double64", "myUID111")
    outlet = StreamOutlet(info, chunk_size=chunk_size, max_buffered=4096*4)

    if y is not None:
        info2 = StreamInfo('y_stream', '', 1, fs, "double64", "myUID112")
        outlet2 = StreamOutlet(info2, chunk_size=chunk_size, max_buffered=4096*4)


    print("sending data now ...")

    #inlet = StreamInlet(info, max_buflen=4096*4)

    # samples= np.empty((0))

    while True:
        
        for n in range(data.shape[-1]//chunk_size):
            
            # samples_old = samples.copy()
            samples = data[:,int(n*chunk_size):int((n+1)*chunk_size)].T
            

            #import pdb; pdb.set_trace()

            outlet.push_chunk(samples.tolist())


            # if n == 30:
            #     inlet = StreamInlet(info, max_buflen=4096*4)

            # if n > 35:
            #     x,ts = inlet.pull_chunk(0.0, max_samples=4096*10)
            #     if ts and n > 45:
            #         x = np.asarray(x)

            #         print(np.array_equal(x,samples))

            if y is not None:
                outlet2.push_chunk(y[int(n*chunk_size):int((n+1)*chunk_size)].tolist())

            time.sleep(chunk_size / fs)


def random_lfps():
    """Generate uniformally distributed random samples and send them out as LSL data stream"""

    fs = 4096.0
    name = "Test Stream"
    type_ = "LFPs"
    n_chan = 14

    chunk_size = 32

    info = StreamInfo(name, type_, n_chan, fs, "double64", "myUID111")
    outlet = StreamOutlet(info, chunk_size=chunk_size, max_buffered=4096*4)

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

    info = StreamInfo(name, type_, n_chan, fs, "double64", "myUID00001")
    outlet = StreamOutlet(info, max_buffered=4096*4)

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
