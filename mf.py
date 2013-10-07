#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import wave
import MFCC
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.lda import LDA
import os
import pickle
import re

SAMPLES_DIR = "samples"


class features_mapper:
    def __init__(self, n_clusters, centers):
        self.k_means = KMeans(n_clusters=n_clusters, max_iter=1, init=centers)

    def map(self, mfcc):
        self.k_means.fit(mfcc)
        new_centers = self.k_means.cluster_centers_
        flat_new_centers = new_centers.flatten()
        return flat_new_centers


def plot_mfcc(mfcc):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = mfcc[:, 0]
    y = mfcc[:, 1]
    points = mfcc[:, 2:4]
    color = np.sqrt((points ** 2).sum(axis=1)) / np.sqrt(2.0)
    rgb = plt.get_cmap('jet')(color)
    ax.scatter(x, y, color=rgb)
    plt.show()
    #plt.savefig("out.png", dpi = 72)


def produce_mfcc(filename, sz):
    wav = wave.open(filename, "r")
    (nchannels, sampwidth, framerate, nframes,
     comptype, compname) = wav.getparams()
    #sz = 44100
    x = np.fromstring(wav.readframes(sz), dtype=np.int16)
    mfcc = MFCC.extract(x)
    return mfcc


def main():
    sz = 22050
    # TODO: rewrite lab_extractor
    lab_extractor = re.compile("(.*)")

    mfcc_list = []
    labels = []
    for filename in os.listdir(SAMPLES_DIR):
        mfcc = produce_mfcc(SAMPLES_DIR + "/" + filename, sz)
        mfcc_list.append(mfcc)
        match = lab_extractor.match(filename)
        labels.append(match.group(1))
        try:
            full_training_sample = \
                np.vstack([full_training_sample, mfcc])
        except:
            full_training_sample = mfcc

    k_means = KMeans(n_clusters=10)
    k_means.fit(full_training_sample)
    centers = k_means.cluster_centers_
    with open("centers", "w") as f:
        pickle.dump(centers, f)

    f_mapper = features_mapper(10, centers)
    for mfcc in mfcc_list:
        new_line = f_mapper.map(mfcc)
        try:
            data = \
                np.vstack([data, new_line])
        except:
            data = new_line

    labels = np.array(labels)
    model = LDA()
    # TODO: prepare proper sample before fitting
    #model.fit(X=data, y=labels)
    #plot_mfcc(mfcc)


if __name__ == '__main__':
    main()
