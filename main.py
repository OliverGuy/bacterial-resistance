
import datetime
import os
import json

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress tf info-level logs
# imports for Keras
import tensorflow as tf

from model import CNNModel, Resizing1D
from preprocessing import classes
from dataset import nucleotides, load_dataset
from plotting import plot_metrics
# from resuming import resume


def main():

    # a few hard-coded values
    voc_size = len(nucleotides)   # 5:NACGT
    batch_size = 16
    epochs = 500
    n_folds = 10
    # change to None for pseudo-random number generation initialized with time:
    random_state = 42
    # set only if fold is not None:
    starting_fold = 0
    # number of parallel workers for data preprocessing
    dataset_parallel_transformations = tf.data.AUTOTUNE
    contig_folder = "../SA-contigs"
    output_root = "../out"
    # tells tensorflow to not reserve all of your GPU for itself:
    memory_growth = False
    output_folder = os.path.join(
        output_root,
        f"checkpoints-{random_state}"
    )

    # cf. https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    if memory_growth:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    # make plots bigger
    matplotlib.rcParams['figure.figsize'] = (12, 10)

    # create folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # data pre-processing
    # prepare data in the correct format
    print("Reading data...")
    ast_data = pd.read_csv(os.path.join(contig_folder, "ast.csv"),
                           header=0,
                           index_col=0)

    antibiotic = "erythromycin"

    # keep only data relative to the chosen antibiotic
    ast_data = ast_data.loc[:, ["contig_path", antibiotic]]
    ast_data.dropna(axis="index", inplace=True)

    X = os.path.join(contig_folder, '') + ast_data["contig_path"].to_numpy()

    # integer-encode classes
    y = ast_data[antibiotic].replace(classes).to_numpy()

    n_classes = len(np.unique(y))  # 2
    print(f"Considering response to {antibiotic}")
    print(f"Number of samples: {ast_data.shape[0]}, {voc_size} nucleotides")
    print(f"{n_classes} unique classes: {np.unique(y)}")

    # Compute class weights to alleviate dataset imbalance
    # scale to keep sum of weights over all samples = y.shape[0]
    class_weights = [y.shape[0] / (n_classes * (y == i).sum())
                     for i in range(n_classes)]

    #compute the expected bias over the whole dataset, since individual folds whill have the same 
    bias_init = y.sum()/y.shape[0]

    for idx, w in enumerate(class_weights):
        print(f"Weight for class {idx}: {w}")

    # dataset parameters
    dataset_params = {
        "n_classes": n_classes,
        "batch_size": batch_size,
        "shuffle": True,
        "random_state": random_state,
        "n_parallel_calls": dataset_parallel_transformations
    }

    # create the network
    def create_network():
        print("Creating network...")

        network = CNNModel(voc_size=voc_size, n_classes=n_classes, bias_init=bias_init)

        # instantiate the optimizer, with its learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

        # the loss is categorical crossentropy, as this is a classification problem
        # TODO use binary crossentropy for multi-label classification and account for unknown classes
        network.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        loss_weights=class_weights,
                        # TODO can be increased to reduce python overhead
                        # once backpropagation gets fixed on GPU:
                        steps_per_execution=1,
                        weighted_metrics=['categorical_accuracy'])

        print("Building network...")
        # calling model to build it

        network.predict(
            load_dataset(X[:batch_size], y[:batch_size],
                         **dataset_params).take(1)
        )
        return network

    network = create_network()

    # freeze the layer to keep embeddings constant:
    # network.get_layer("embedding").trainable = False

    # starting_epoch = 0

    initial_weights_path = os.path.join(
        output_folder, "initial-random-weights.h5")
    if (random_state is None) or not os.path.exists(initial_weights_path):
        print("No initial weights found, or random_state not set.")

        # save the initial random weights of the network, to reset them
        # later before each fold
        print(f"Saving new random weights at {initial_weights_path}")
        network.save_weights(initial_weights_path)

    # TODO use this to enable resuming once whole model saving and
    # loading gets fixed for subclassed models:

    # else:
    #     starting_fold, starting_epoch, checkpoint_path = resume(
    #         output_folder, epochs)
    #     print(
    #         f"Resuming training at fold {starting_fold + 1}, epoch {starting_epoch}")
    #     if checkpoint_path is not None:
    #         print(f"Restoring from checkpoint: {checkpoint_path}")
    #         with tf.keras.utils.custom_object_scope({
    #             "CNNModel": CNNModel,
    #             "Resizing1D": Resizing1D
    #         }):
    #             network = tf.keras.models.load_model(
    #                 checkpoint_path
    #             )

    network.summary()

    # a 'callback' in Keras is a condition that is monitored during the training process
    # here we instantiate a callback for an early stop, that is used to avoid overfitting
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-5,
        patience=100,
        verbose=1,
        restore_best_weights=True  # to evaluate according to the best settings found
    )

    #  cross-validation
    # TODO adapt stratification to multi-label classification
    from sklearn.model_selection import StratifiedShuffleSplit
    stratified_shuffle_split = StratifiedShuffleSplit(n_splits=n_folds,
                                                      test_size=0.1,
                                                      random_state=random_state)

    # this method needs numeric labels for y, but does not check data from X
    for fold, (train_and_val_index, test_index) in enumerate(stratified_shuffle_split.split(X, y)):

        # skip folds until 'starting_fold', used to stop and restart evaluations
        if fold < starting_fold:
            continue

        # stratified k-fold only splits the data in two, so training and validation are together
        X_train_and_val, X_test = X[train_and_val_index], X[test_index]
        y_train_and_val, y_test = y[train_and_val_index], y[test_index]

        # get test and validation set indexes, using a StratifiedShuffleSplit with just one split
        validation_shuffle_split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=random_state)
        train_index, val_index = next(validation_shuffle_split.split(
            X_train_and_val, y[train_and_val_index]))

        X_train, X_val = X_train_and_val[train_index], X_train_and_val[val_index]
        y_train, y_val = y_train_and_val[train_index], y_train_and_val[val_index]

        fold_report = f"Fold {fold+1}/{n_folds} (samples train={len(X_train)}, validation={len(X_val)}, test={len(X_test)})"

        print(fold_report + ": starting the training process...")

        training_dataset = load_dataset(X_train, y_train, **dataset_params)
        validation_dataset = load_dataset(X_val, y_val, **dataset_params)
        testing_dataset = load_dataset(X_test, y_test, **dataset_params)
        # reset network to initial state
        if fold != 0:
            network = create_network()  # recreate network to reset optimizer state
        network.load_weights(os.path.join(
            output_folder, "initial-random-weights.h5"))

        fold_folder = os.path.join(output_folder, f"fold-{fold}")

        os.makedirs(fold_folder, exist_ok=True)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(fold_folder,
                         r"epoch-{epoch:03d}-{val_loss:.3f}-{val_categorical_accuracy:.3f}"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            # True is equivalent to `model.save_weights`,
            # False is equivalent to `model.save`
            mode="min",
            save_freq="epoch"
        )

        # TODO break up epochs to give finer control to early stopping ?
        # if so, maybe change checkpoint frequency
        train_history = network.fit(
            training_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            # initial_epoch=starting_epoch,
            callbacks=[
                early_stopping_callback,
                checkpoint_callback
            ]
        )
        # see generator_params

        print("Training process finished. Testing...")
        test_history = network.evaluate(
            testing_dataset
        )

        print("Dumping training history...")
        with open(os.path.join(fold_folder, "history.json"), 'w') as fp:
            json.dump(train_history.history, fp)

        plot_metrics(network.metrics_names, train_history)
        plt.savefig(os.path.join(fold_folder, "metrics.png"))

        train_accuracy = train_history.history["categorical_accuracy"][-1]
        val_accuracy = train_history.history["val_categorical_accuracy"][-1]
        accuracy_idx = network.metrics_names.index('categorical_accuracy')
        test_accuracy = test_history[accuracy_idx]

        accuracy_report = f"Accuracy on training: {train_accuracy:.4f}, validation: {val_accuracy:.4f}, test: {test_accuracy:.4f}"

        print(accuracy_report)

        # save everything to a folder: fold; predictions on training, test, validation; model
        print("Saving information for the current fold...")

        # save model (divided in two parts: network layout and weights)
        network_json = network.to_json()
        with open(os.path.join(fold_folder, f"model.json"), "w") as fp:
            fp.write(network_json)
        network.save_weights(os.path.join(fold_folder, f"weights"))

        # save information about the fold
        with open(os.path.join(output_folder, "global-summary.txt"), "a") as fp:
            fp.write(fold_report)
            fp.write(accuracy_report)

    return


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()
