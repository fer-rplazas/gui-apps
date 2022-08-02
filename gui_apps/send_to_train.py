import os
from pathlib import Path
from typing import Union


def send_data_to_jade(
    source: Union[str, os.PathLike],
    destination: Union[str, os.PathLike] = "online_runs/datafiles",
):
    """Upload file to jade at given destination"""

    send_cmd = f"pscp -load jade {source} fernando@jade.mrc.ox.ac.uk:{destination}"
    os.system(send_cmd)


def train_and_return(local_path: Union[str, os.PathLike]):
    """Run training script on jade for a given dataset and return resulting models to local dir"""

    dataset_filename = Path(Path(local_path).name[:-1]).with_suffix(".h5")

    suffixes = [".cnn", ".svm", ".pkl"]

    train_cmd = f"""plink -load jade -batch "cd ~/online_runs/ && /home/fernando/anaconda3/envs/all-v4/bin/python train_v2.py --file datafiles/{str(dataset_filename)}" """
    get_cmds = [
        f"pscp -load jade fernando@jade.mrc.ox.ac.uk:online_runs/datafiles/{str(Path(dataset_filename).with_suffix(sfx))} {str(Path(local_path).with_suffix(sfx))}"
        for sfx in suffixes
    ]

    os.system(train_cmd)
    [os.system(get_cmd) for get_cmd in get_cmds]
