#!/usr/bin/python
# -*- coding: utf-8 -*-

from mf import get_arguments
import mfcc_utils
import pickle
import numpy as np

MODEL_FILE_NAME = "model"
CENTERS_FILE_NAME = "centers"


def main():
    args = get_arguments()
    mfcc_list, labels = mfcc_utils.read_files_from_dir(
        dir_name=args.dir, processes=args.processes, sz=args.sz)
    with open(MODEL_FILE_NAME, "r") as f:
        model = pickle.load(f)
    with open(CENTERS_FILE_NAME, "r") as f:
        centers = pickle.load(f)
    labels = np.array(labels)
    # learning new centers for each wav file
    f_mapper = mfcc_utils.features_mapper(args.centers, centers)
    features_list = map(f_mapper.map, mfcc_list)

    # generating training sample
    data = mfcc_utils.to_stack(features_list)
    predictions = model.predict(X=data)
    print predictions


if __name__ == "__main__":
    main()
