"""Multiprocessing-friendly `keras.utils.Sequence`-based data loader for 
contigs and associated AST data.
Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

import os

import numpy as np
import keras

from numpy.core.numeric import full

from contigParser import parsers


class ContigDataGenerator(keras.utils.Sequence):
    """Generates data to fit Keras models on.

    Parameters
    ----------
    ids : array-like of hashables
        Index shared by `paths` and `labels`, e.g. the run accession when the 
        two are taken from the ast DataFrame.
    paths : array-like of path-likes
        Paths to the contig files, relative to the specified `folder`.
    labels : array-like of ints
        Classes corresponding to the contigs.
    folder : path-like
        Path to the contigs folder; individual contig paths are computed by 
        joining `folder` and `paths[i]`.
    dim : int, optional
        The max length of the contigs.
    batch_size : int, default: 32
        The length of each individual batch.
    n_channels : int, default: 1
        The number of channels in the output.
    n_classes : int, optional
        The number of classes; default (None) is inferred as `max(labels) + 1` 
        by `keras.utils.to_categorical`.
    shuffle : bool, default: True
        Whether to shuffle the dataset before each epoch.
    """

    def __init__(self, ids, paths, labels, folder, parser="max", batch_size=32, dim=None, n_channels=1, n_classes=None, shuffle=True):
        self.ids = ids
        self.paths = paths
        self.dim = dim
        self.labels = labels
        self.folder = folder
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.parser = parsers[parser]
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        "Generates one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch (called by Keras)."
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, dim, n_channels)
        "Generates data containing batch_size samples"
        # Initialization
        sequences = []
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, id in enumerate(list_IDs_temp):
            fullpath = os.path.join(self.folder, self.paths[id])
            sequences.append(self.parser(id, fullpath))

            y[i] = self.labels[id]

        # NB: padding is pre by default
        X = keras.preprocessing.pad_sequences(sequences, maxlen=self.dim)
        # TODO masking ?
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
