import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet, local_clock
import random


def main():
    fs = 4096.0
    name = "Test Stream"
    type = "LFPs"
    n_chan = 14

    chunk_size = 32

    info = StreamInfo(name, type, n_chan, fs, "float32", "myUID111")

    outlet = StreamOutlet(info, chunk_size=chunk_size)

    print("sending data now ...")
    start_time = local_clock()
    sent_chunks = 0

    while True:
        required_samples = chunk_size

        samples = [
            (np.random.rand(required_samples) + 10_000).tolist() for _ in range(n_chan)
        ]

        outlet.push_chunk(samples)
        sent_chunks += 1

        time.sleep(chunk_size / fs)


def main_control():
    fs = 2
    name = "simulated_cnn_output"
    type = "control_signal"
    n_chan = 2

    info = StreamInfo(name, type, n_chan, fs, "float32", "myUID00001")
    outlet = StreamOutlet(info)

    print("sending data now...")
    sent = False
    true_opts = [True] * 24
    true_opts.append(False)

    false_opts = [False] * 24
    false_opts.append(True)

    while True:

        if sent == True:
            sample = random.choice(true_opts)

        else:
            sample = random.choice(false_opts)

        sent = sample
        samples = [sample, sample]

        outlet.push_sample(samples)
        time.sleep(1.0 / fs)


if __name__ == "__main__":
    main_control()
