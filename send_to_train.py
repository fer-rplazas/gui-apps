import os
from pathlib import Path


def send(source, destination="temp/"):

    send_cmd = f"pscp -load jade {source} fernando@jade.mrc.ox.ac.uk:{destination}"

    os.system(send_cmd)


def train_and_return(destination_local, **kwargs):

    fname_train = Path(Path(destination_local).name[:-1]).with_suffix(".h5")
    train_cmd = f"""plink -load jade -batch "cd ~/temp/ && /home/fernando/anaconda3/envs/all-v4/bin/python train.py --file {str(fname_train)}" """
    get_cmd = f"pscp -load jade fernando@jade.mrc.ox.ac.uk:temp/{str(Path(fname_train).with_suffix('.pt'))} {str(Path(destination_local).with_suffix('.pt'))}"

    os.system(train_cmd)
    os.system(get_cmd)


if __name__ == "__main__":
    pass
