
import datetime
import os
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress tf info-level logs
# imports for Keras
import tensorflow as tf
from model import CNNModel

from preprocessing import classes
from contigParser import nucleotides

from contigDataset import ContigDataGenerator


def main():

    # a few hard-coded values
    voc_size = len(nucleotides)   # 5:NACGT
    batch_size = 16
    epochs = 500
    n_folds = 10
    parser = "cut"  # cf. contigParser.py
    # this is used to stop and restart the testing on a sequence of folds
    # (only works with a fixed random state)
    starting_fold = 0
    # TODO change to None for pseudo-random number generation initialized with time:
    random_state = 42
    gen_multiprocessing = False
    gen_workers = 1
    contig_folder = "../SA-contigs"
    output_root = "../out"
    # tells tensorflow to not reserve all of your GPU for itself:
    memory_growth = False
    output_folder = os.path.join(
        output_root,
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-keras-cnn-output"
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

    # create folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # data pre-processing
    # prepare data in the correct format
    print("Reading data...")
    ast_data = pd.read_csv(os.path.join(contig_folder, "ast.csv"),
                           header=0,
                           index_col=0)

    antibiotic = "gentamicin"

    # keep only data relative to the chosen antibiotic
    ast_data = ast_data.loc[:, ["contig_path", antibiotic]]
    ast_data.dropna(axis="index", inplace=True)

    X = ast_data["contig_path"].to_numpy()

    # integer-encode classes
    y = ast_data[antibiotic].replace(classes).to_numpy()

    n_classes = len(np.unique(y))  # 2
    print(f"Considering response to {antibiotic}")
    print(f"Number of samples: {ast_data.shape[0]}, {voc_size} nucleotides")
    print(f"{n_classes} unique classes: {np.unique(y)}")

    # dataset generator parameters
    generator_params = {
        "folder": contig_folder,
        "n_classes": n_classes,
        "parser": parser,
        "batch_size": batch_size,
        "shuffle": True,
        "random_state": random_state
    }

    # create the network
    print("Creating network...")

    network = CNNModel(voc_size=voc_size, n_classes=n_classes)

    # instantiate the optimizer, with its learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # the loss is categorical crossentropy, as this is a classification problem
    # TODO use binary crossentropy for multi-label classification and account for unknown classes
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])

    print("Building network...")
    # calling model to build it

    network.predict(ContigDataGenerator(X[:batch_size],
                                        y[:batch_size],
                                        **generator_params))

    # freeze the layer to keep embeddings constant:
    # network.get_layer("embedding").trainable = False
    network.summary()

    # a 'callback' in Keras is a condition that is monitored during the training process
    # here we instantiate a callback for an early stop, that is used to avoid overfitting
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-5,
        patience=100,
        verbose=1,
        restore_best_weights=False)

    # before going into the cross-validation, we actually save the initial
    # random weights of the network, to reset it later before each fold
    network.save_weights(os.path.join(
        output_folder, "initial-random-weights.h5"))

    #  cross-validation
    # TODO adapt stratification to multi-label classification
    from sklearn.model_selection import StratifiedShuffleSplit
    stratified_shuffle_split = StratifiedShuffleSplit(n_splits=n_folds,
                                                      test_size=0.1,
                                                      random_state=random_state)

    # this method needs numeric labels fo y, but does not check data from X
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

        fold_report = f"Fold {fold+1}/{n_folds} (samples train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}"

        print(fold_report + ": starting the training process...")

        training_generator = ContigDataGenerator(
            X_train, y_train, **generator_params)
        validation_generator = ContigDataGenerator(
            X_val, y_val, **generator_params)
        testing_generator = ContigDataGenerator(
            X_test, y_test, **generator_params)
        # reset network to initial state
        network.load_weights(os.path.join(
            output_folder, "initial-random-weights.h5"))

        train_history = network.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=epochs,
            workers=gen_workers,
            use_multiprocessing=gen_multiprocessing
        )  # , callbacks=[early_stopping_callback])
        # see generator_params

        test_history = network.evaluate(
            testing_generator,
            epochs=epochs,
            workers=gen_workers,
            use_multiprocessing=gen_multiprocessing
        )

        print("Training process finished. Testing...")
        # TODO to be tested; training doesn't reach that stage in reasonable time yet

        print(train_history.history.keys())

        train_accuracy = train_history.history["categorical_accuracy"]
        val_accuracy = train_history.history["val_categorical_accuracy"]
        test_accuracy = test_history.history["categorical_accuracy"]

        # TODO tensorboard ?

        accuracy_report = f"Accuracy on training: {train_accuracy:.4f}, validation: {val_accuracy:.4f}, test: {test_accuracy:.4f}"

        print(accuracy_report)

        # save everything to a folder: fold; predictions on training, test, validation; model
        print("Saving information for the current fold...")

        # save model (divided in two parts: network layout and weights)
        network_json = network.to_json()
        with open(os.path.join(output_folder, f"fold-{fold}-model.json"), "w") as fp:
            fp.write(network_json)
        network.save_weights(os.path.join(
            output_folder, f"fold-{fold}-weights.h5"))

        # save information about the fold
        with open(os.path.join(output_folder, "global-summary.txt"), "a") as fp:
            fp.write(fold_report)
            fp.write(accuracy_report)

        # save data of the fold
        #x_column_names = ["feature_%d" % f for f in range(0, sequence_length) ]

        # df_train = pd.DataFrame({
        #     "y_true": y_train_labels.reshape(-1),
        #     "y_pred": y_train_pred_labels.reshape(-1)
        # })
        # #for i, c in enumerate(x_column_names) : df_train[c] = X_train[:,0,i,0].reshape(-1)
        # df_train.to_csv(os.path.join(output_folder, f"fold-{fold}-training.csv"), index=False)

        # df_val = pd.DataFrame({
        #     "y_true": y_val_labels.reshape(-1),
        #     "y_pred": y_val_pred_labels.reshape(-1)
        # })
        # df_val.to_csv(os.path.join(output_folder, f"fold-{fold}-validation.csv"), index=False)

        # df_test = pd.DataFrame({
        #     "y_true": y_test_labels.reshape(-1),
        #     "y_pred": y_test_pred_labels.reshape(-1)
        # })
        # df_test.to_csv(os.path.join(output_folder, f"fold-{fold}-test.csv"), index=False)

    return


if __name__ == "__main__":
    main()
