import math

import numpy as np


def split_training(num_images, train_split_percentage, split):
    # filter image_filenames and poses based on train/eval split percentage
    num_train_images = math.ceil(train_split_percentage * num_images)
    num_test_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    # eval images are the remaining images
    i_test = np.setdiff1d(i_all, i_train)
    assert len(i_test) == num_test_images
    if split == "train":
        indices = i_train
    elif split in ["val", "test"]:
        indices = i_test
    elif split == 'all' or split == 'rendering':
        indices = i_all
    else:
        raise ValueError(f"Unknown dataparser split {split}")
    return indices
