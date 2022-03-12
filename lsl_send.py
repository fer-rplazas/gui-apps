import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet, local_clock


def main():
    fs = 4096.0
    name = "dd"
    type = "LFPs"
    n_chan = 11

    chunk_size = 32

    info = StreamInfo(name, type, n_chan, fs, "float32", "myUID111")

    outlet = StreamOutlet(info, chunk_size=chunk_size)

    print("sending data now ...")
    start_time = local_clock()
    sent_chunks = 0

    while True:
        required_samples = chunk_size

        samples = [np.random.rand(required_samples).tolist() for _ in range(n_chan)]

        outlet.push_chunk(samples)
        sent_chunks += 1

        time.sleep(chunk_size / fs)


if __name__ == "__main__":
    main()
