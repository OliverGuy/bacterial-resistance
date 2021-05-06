"""Multiprocessing-friendly `keras.utils.Sequence`-based data loader for
contigs and associated AST data.

Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from contigParser import parsers


class ContigDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, paths, labels, folder, parser="max", batch_size=32, sequence_length=None,
                 n_classes=None, shuffle=True, random_state=None, **parser_kwargs):
        """Generates data to fit Keras models on. Use as an array of 
        batches or in the `.fit` method of a `tf.keras` model.

        Batches will have shape `batch_size * sequence_length`, while true labels will
        have size `batch_size * n_classes`.

        Parameters
        ----------
        paths : array of path-likes
            Paths to the contig files, relative to the specified `folder`.
        labels : array of ints
            Classes corresponding to the contigs.
        folder : path-like
            Path to the contigs folder; individual contig paths are 
            computed by joining `folder` and `paths[i]`.
            `contigParser` Parser to use. Non-string objects will be 
            considered to be already-initialised instances that have an
            `ndims` attribute and implement `__call__`. See `contigParser`
            for details.
        batch_size : int, default: 32
            The length of each individual batch.
        sequence_length : int, optional
            The max length of the parsed contigs.
        n_classes : int, optional
            The number of classes; default (None) is inferred as 
            `max(labels) + 1` by `keras.utils.to_categorical`.
        shuffle : bool, default: True
            Whether to shuffle the dataset before each epoch.
        random_state : int, optional
            The seed to use for shuffling.
        **parser_kwargs : dict, optional
            Keyword arguments to be passed to the parser. Used only if the
            `parser` is a string.
        """
        self.paths = paths
        self.sequence_length = sequence_length
        self.labels = labels
        self.folder = folder
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        if shuffle:
            self.rg = np.random.default_rng(random_state)
        if isinstance(parser, str):
            self.parser = parsers[parser](**parser_kwargs)
        else:
            # the parser is considered a contigParser instance
            self.parser = parser
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return len(self.labels) // self.batch_size

    def __getitem__(self, index):
        """Generates one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        """Updates indexes after each epoch (called by Keras)."""
        if self.shuffle:
            self.indexes = np.arange(len(self.labels))
            self.rg.shuffle(self.indexes)

    def __data_generation(self, temp_IDs):
        # X : (n_samples, dim)
        """Generates data containing batch_size samples."""
        # Initialization
        sequences = []

        # Generate data
        for id in temp_IDs:
            fullpath = os.path.join(self.folder, self.paths[id])
            sequences.append(self.parser(id, fullpath))

        # 'post' padding allows models to use fast CuDNN layer implementations
        X = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.sequence_length, padding='post', truncating='post')
        y = tf.keras.utils.to_categorical(self.labels[temp_IDs], num_classes=self.n_classes)
        # TODO sample_weights ?
        return X, y


if __name__ == "__main__":
    import pandas as pd
    from preprocessing import classes
    from time import perf_counter

    print("loading ast...")

    ast_data = pd.read_csv("../SA-contigs/ast.csv", header=0, index_col=0)

    antibiotic = "gentamicin"

    # keep only data relative to the chosen antibiotic
    ast_data = ast_data.loc[:, ["contig_path", antibiotic]]
    ast_data.dropna(axis="index", inplace=True)

    X = ast_data["contig_path"].to_numpy()

    # integer-encode classes
    y = ast_data[antibiotic].replace(classes).to_numpy()

    params = {
        "folder": "../SA-contigs",
        "n_classes": 2,
        "parser": "max",
        "batch_size": 32,
        "shuffle": True
    }

    print("testing generator...")

    gen = ContigDataGenerator(X, y, **params)
    t = perf_counter()
    X_batch, y_batch = gen[0]
    t = perf_counter() - t
    print(f"random batch generated in {t} s !")
    print(f"X_batch shape: {X_batch.shape}, Y_batch shape: {y_batch.shape}")

    print("X.head:")
    print(X_batch[:5])
    print("y.head:")
    print(y_batch[:5])
