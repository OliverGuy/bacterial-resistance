# TODO move to preprocessing

import os
import sys

import numpy as np
import tensorflow as tf

nucleotides = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}

init = tf.lookup.KeyValueTensorInitializer(
    keys=tf.constant(list(nucleotides.keys())),
    values=tf.constant(list(nucleotides.values()), dtype=tf.int64)
)
table = tf.lookup.StaticVocabularyTable(
    init,
    num_oov_buckets=1
)


def parse_fasta(filename, label):
    # HACK
    tf.print(filename, output_stream="file://../tmp/file_list.txt")
    contig = tf.io.read_file(filename)
    # delete the node names:
    contig = tf.strings.regex_replace(contig, r">.*\n", ">")
    # remove whitespaces i.e. [\t\n\f\r ]
    contig = tf.strings.regex_replace(contig, r"\s", "")
    # separate the contig into sequences
    # (the first line would have been empty)
    contig = tf.strings.split(contig, ">")[1:]
    # separate the sequences into individual nucleotides:
    contig = tf.strings.bytes_split(contig)
    # embed the nucleotides into integers:
    contig = table.lookup(contig)
    # cast the tensor to int8 to save memory:
    contig = tf.cast(contig, tf.int8)
    return contig, label


def load_dataset(paths, labels, n_classes, batch_size=32, shuffle=True,
                 random_state=None, n_parallel_calls=None, n_prefetch=2):
    # integer-encode classes:
    if n_classes is None:
        n_classes = len(np.unique(labels))
    y = tf.keras.utils.to_categorical(labels, num_classes=n_classes)

    dataset = tf.data.Dataset.from_tensor_slices((paths, y))
    # shuffle before parsing, as you would otherwise have to wait (around
    # 15 minutes) for the whole dataset to be parsed:
    if shuffle:
        dataset = dataset.shuffle(
            dataset.cardinality(),
            seed=random_state,
            reshuffle_each_iteration=True
        )
    dataset = dataset.map(parse_fasta, num_parallel_calls=n_parallel_calls)
    dataset = dataset.batch(batch_size)
    # HACK
    # dataset = dataset.prefetch(n_prefetch)
    return dataset


def __load_all():
    """Returns a dataset with a few default parameters, useful if you want
    to toy with the dataset in jupyter """
    import pandas as pd
    from preprocessing import classes

    print("loading ast...")

    contig_folder = r"../SA-contigs"

    ast_data = pd.read_csv(os.path.join(contig_folder, "ast.csv"),
                           header=0, index_col=0)

    antibiotic = "gentamicin"

    # keep only data relative to the chosen antibiotic
    ast_data = ast_data.loc[:, ["contig_path", antibiotic]]
    ast_data.dropna(axis="index", inplace=True)

    X = os.path.join(contig_folder, '') + ast_data["contig_path"].to_numpy()

    # integer-encode classes
    y = ast_data[antibiotic].replace(classes).to_numpy()

    print("loading dataset...")

    dataset_params = {
        "n_classes": 2,
        "batch_size": 16,
        "shuffle": True,
        "random_state": 42,
        "n_parallel_calls": tf.data.AUTOTUNE
    }

    return load_dataset(X, y, **dataset_params)


if __name__ == "__main__":
    from time import perf_counter
    dataset = __load_all()

    print(dataset.cardinality())

    print(dataset.element_spec)

    print("Inspecting a data point:")

    t = perf_counter()
    for elem in dataset.take(1):
        t = perf_counter() - t
        print(f"batch generated in {t} s !")
        for feature in elem:
            print(type(feature), feature.shape)
        t, y = elem
        print(t.row_lengths(axis=0))
        for idx, subtensor in enumerate(t):
            print(idx)
