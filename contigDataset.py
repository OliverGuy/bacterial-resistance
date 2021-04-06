"""Multiprocessing-friendly `keras.utils.Sequence`-based data loader for
contigs and associated AST data.

Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

import os

import numpy as np
import keras

from contigParser import parsers


class ContigDataGenerator(keras.utils.Sequence):

    def __init__(self, paths, labels, folder, parser="max", batch_size=32, dim=None, n_classes=None, shuffle=True):
        """Generates data to fit Keras models on.

        Batches will have shape `batch_size`, while true labels will 
        have size `batch_size * n_classes`.

        Parameters
        ----------
        paths : array-like of path-likes
            Paths to the contig files, relative to the specified `folder`.
        labels : array-like of ints
            Classes corresponding to the contigs.
        folder : path-like
            Path to the contigs folder; individual contig paths are 
            computed by joining `folder` and `paths[i]`.
        parser: {'max'}
            `contigParser` Parser to use.
        batch_size : int, default: 32
            The length of each individual batch.
        dim : int, optional
            The max length of the contigs.
        n_classes : int, optional
            The number of classes; default (None) is inferred as 
            `max(labels) + 1` by `keras.utils.to_categorical`.
        shuffle : bool, default: True
            Whether to shuffle the dataset before each epoch.
        """
        # TODO random state, initial state, check docstrings
        self.paths = paths
        self.dim = dim
        self.labels = labels
        self.folder = folder
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.parser = parsers[parser]
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch (called by Keras)."""
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, dim)
        """Generates data containing batch_size samples."""
        # Initialization
        sequences = []
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, id in enumerate(list_IDs_temp):
            fullpath = os.path.join(self.folder, self.paths[id])
            sequences.append(self.parser(id, fullpath))

            y[i] = self.labels[id]

        # 'post' padding allows models to use fast CuDNN layer implementations
        X = keras.preprocessing.pad_sequences(sequences, maxlen=self.dim, padding='post')
        # TODO masking ?
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
