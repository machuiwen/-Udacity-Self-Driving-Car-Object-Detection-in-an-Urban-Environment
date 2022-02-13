import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    files = [f for f in glob.glob(data_dir + '/training_and_validation/*.tfrecord')]
    random.shuffle(files)

    split_ratio = 0.8
    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]

    train_dir = os.path.join(data_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(data_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)

    for f in train_files:
        f_base = os.path.basename(f)
        os.rename(f, os.path.join(train_dir, f_base))

    for f in val_files:
        f_base = os.path.basename(f)
        os.rename(f, os.path.join(val_dir, f_base))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)