import wave
import MFCC
import numpy as np
import multiprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re
import os
import sys


lab_extractor = re.compile("([^\-]*)\-")


def unwrap_produce_mfcc(args):
    return args[0].produce_mfcc(args[1])


class mfcc_reader:
    def __init__(self, lab_extractor, sz, num_workers=1):
        self.lab_extractor = lab_extractor
        self.sz = sz
        self.num_workers = num_workers

    def produce_mfcc(self, filename):
        wav = wave.open(filename, "r")
        x = np.fromstring(wav.readframes(self.sz), dtype=np.int16)
        #(nchannels, sampwidth, framerate, nframes,
        # comptype, compname) = wav.getparams()
        mfcc = MFCC.extract(x)
        match = self.lab_extractor.match(filename)
        try:
            label = match.group(1)
        except:
            label = "unknown"
            print >> sys.stderr, "unknown labels encountered"
        return (mfcc, label)

    def read_list(self, file_list):
        pool = multiprocessing.Pool(self.num_workers)
        return pool.map(unwrap_produce_mfcc,
                        zip([self] * len(file_list), file_list))


class features_mapper:
    def __init__(self, n_clusters, centers):
        self.k_means = KMeans(n_clusters=n_clusters, max_iter=1, init=centers)

    def map(self, mfcc):
        self.k_means.fit(mfcc)
        new_centers = self.k_means.cluster_centers_
        flat_new_centers = new_centers.flatten()
        return flat_new_centers


def to_stack(data_list):
    for item in data_list:
        try:
            result = np.vstack([result, item])
        except:
            result = item
    return result


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


def get_file_list(dir_name):
    return map(lambda x: dir_name + "/" + x, os.listdir(dir_name))


def read_files_from_dir(dir_name, processes, sz):
    file_list = get_file_list(dir_name)

    reader = mfcc_reader(lab_extractor,
                         sz, num_workers=processes)
    mfcc_tuples_list = reader.read_list(file_list)

    mfcc_list = map(lambda x: x[0], mfcc_tuples_list)
    labels = map(lambda x: x[1], mfcc_tuples_list)
    return mfcc_list, labels
