#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
import pickle
import mfcc_utils
import argparse


# magic constants
CENTERS_FILE_NAME = "centers"
DATA_FILE_NAME = "data"
LABELS_FILE_NAME = "labels"


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--processes", default=1,
                        help="number of processes, default=1", type=int)
    parser.add_argument("-d", "--dir", default="samples",
                        help="directory with .wav files, default=samples")
    parser.add_argument("-s", "--sz", default=44100, type=int,
                        help="specify sz, default=44100")
    parser.add_argument("-c", "--centers", default=10,
                        help="number of centers", type=int)
    args = parser.parse_args()
    return args


def main():
    # parsing arguments
    args = get_arguments()
    # reading files and transforming them into mfcc format
    mfcc_list, labels = mfcc_utils.read_files_from_dir(
        dir_name=args.dir, processes=args.processes, sz=args.sz)
    labels = np.array(labels)
    full_training_sample = mfcc_utils.to_stack(mfcc_list)

    # learning overall centers
    k_means = KMeans(n_clusters=args.centers)
    k_means.fit(full_training_sample)
    centers = k_means.cluster_centers_
    with open(CENTERS_FILE_NAME, "w") as f:
        pickle.dump(centers, f)

    # learning new centers for each wav file
    f_mapper = mfcc_utils.features_mapper(args.centers, centers)
    features_list = map(f_mapper.map, mfcc_list)

    # generating training sample
    data = mfcc_utils.to_stack(features_list)

    # saving everything
    with open(DATA_FILE_NAME, "w") as f:
        pickle.dump(data, f)

    with open(LABELS_FILE_NAME, "w") as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    main()
