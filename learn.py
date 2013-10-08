#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix


def main():
    with open("data", "r") as f:
        data = pickle.load(f)

    with open("labels", "r") as f:
        labels = pickle.load(f)

    model = LDA()
    model.fit(X=data, y=labels)
    predictions = model.predict(data)
    conf_matr = confusion_matrix(y_true=labels, y_pred=predictions)
    print conf_matr



if __name__ == '__main__':
    main()
