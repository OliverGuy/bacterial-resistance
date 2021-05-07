
import h5py  # this is used to save Keras models in the hdf5 format
import datetime
import os
import numpy as np
import pandas as pd
import sys

# imports for Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Flatten, MaxPooling1D, Embedding

from preprocessing import classes
from contigParser import nucleotides

from contigDataset import ContigDataGenerator


def main():

    # a few hard-coded values
    sequence_length = 1299315  # length of a DNA/RNA sequence for the bacteria, None=inferred
    voc_size = len(nucleotides)  # 5:ACGTN
    batch_size = 50
    epochs = 1000  # 500
    n_folds = 10
    parser = "cut"  # cf. contigParser.py
    starting_fold = 0
    random_state = 42  # TODO change to None for pseudo-random number generation initialized with time
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

    log_dir = "../logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # create folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # data pre-processing
    # prepare data in the correct format
    print("Reading data...")
    ast_data = pd.read_csv(os.path.join(contig_folder, "ast.csv"), header=0, index_col=0)

    antibiotic = "gentamicin"

    # keep only data relative to the chosen antibiotic
    ast_data = ast_data.loc[:, ["contig_path", antibiotic]]
    ast_data.dropna(axis="index", inplace=True)

    X = ast_data["contig_path"].to_numpy()

    # integer-encode classes
    y = ast_data[antibiotic].replace(classes).to_numpy()

    number_of_classes = len(np.unique(y))  # 2
    print(f"Considering response to {antibiotic}")
    print(f"Number of samples: {ast_data.shape[0]}, {voc_size} nucleotides")
    print(f"{number_of_classes} unique classes: {np.unique(y)}")

    # dataset generator parameters
    generator_params = {
        "folder": contig_folder,
        "parser": parser,
        "batch_size": batch_size,
        "shuffle": True,
        "random_state": random_state
    }
    # length of output contigs must be consistent for the dense layer
    if not sequence_length:
        sequence_length = ContigDataGenerator(X, y, **generator_params).compute_sequence_length()
    generator_params["sequence_length"] = sequence_length
    print(f"sequence length: {sequence_length}")

    # shapes of the tensors in the network
    # (?, 1299315, 1)
    # (?, 210, 130)
    # (?, 1, 204)
    # (?, 1, 150)
    # (?, 5)

    # input
    # python3 gen3.py 500 130 204 150 196 148 236 81 9 106 121 0 0 0

    # default values for the network
    embed_dim = 1
    # in order NACGT, see nucleotides:
    embed_init_w = np.array([0, 1, 0.25, 0.75, 0.5]).reshape((-1, 1))

    units_1 = 12  # 130
    units_2 = 204
    units_3 = 150
    units_4 = 196
    pool_size_1 = 148
    pool_size_2 = 236
    pool_size_3 = 81
    conv_window_length_1 = 21
    conv_window_length_2 = 106
    conv_window_length_3 = 121
    dropout_probability = 0.5

    # default value for bias
    bias_vector_init_value = 0.1

    # the loss is adjusted with a parameter for l2 regularization in each of
    # the convolutional layers, called "beta" in the gen3.py script
    penalty_regularization_w_conv = 0.001

    # create the network
    print("Creating network...")
    network = Sequential()

    network.add(Input(
        shape=(sequence_length,)
    ))
    print(network.input_shape)
    # embeds integral indices into dense arrays
    # TODO also try with a one-hot encoding
    embed_layer = Embedding(voc_size, embed_dim)
    network.add(embed_layer)
    network.get_layer("embedding").set_weights([embed_init_w])
    # freeze the model to keep embeddings constant:
    # network.get_layer("embedding").trainable = False
    print(network.output_shape)


    # using conv1D instead of resizing the data to make it fit conv2D
    # 1 Layer
    network.add(Conv1D(
        units_1,
        activation='relu',
        kernel_size=conv_window_length_1,
        padding='same',
        use_bias=True,
        bias_initializer=tf.keras.initializers.Constant(bias_vector_init_value),
        kernel_regularizer=tf.keras.regularizers.l2(penalty_regularization_w_conv)))
    print(network.output_shape)
    network.add(MaxPooling1D(pool_size=pool_size_1, padding='same'))
    print(network.output_shape)

    # 2 Layer
    #network.add(Conv1D(units_2, activation='relu', kernel_size=conv_window_length_2, padding='same', use_bias=True, bias_initializer=keras.initializers.Constant(bias_vector_init_value), kernel_regularizer=keras.regularizers.l2(penalty_regularization_w_conv)))
    #network.add(MaxPooling2D(pool_size=pool_size_2, padding='same'))

    # 3 Layer
    #network.add(Conv1D(units_3, activation='relu', kernel_size=conv_window_length_3, padding='same', use_bias=True, bias_initializer=keras.initializers.Constant(bias_vector_init_value), kernel_regularizer=keras.regularizers.l2(penalty_regularization_w_conv)))
    #network.add(MaxPooling2D(pool_size=pool_size_3, padding='same'))

    # Rectifier Layer, with dropout
    network.add(Flatten())  # convert tensor in N dimensions to tensor in 2
    print(network.output_shape)
    network.add(Dense(
        units_4,
        activation='relu',
        use_bias=True,
        bias_initializer=tf.keras.initializers.Constant(bias_vector_init_value)))
    network.add(Dropout(dropout_probability))
    print(network.output_shape)

    # (Output) Softmax
    network.add(Dense(
        number_of_classes,
        activation='softmax',
        use_bias=True,
        bias_initializer=tf.keras.initializers.Constant(bias_vector_init_value)
    ))
    print(network.output_shape)

    # instantiate the optimizer, with its learning rate
    from keras.optimizers import Adam
    optimizer = Adam(learning_rate=1e-5)

    # the loss is categorical crossentropy, as this is a classification problem
    # TODO use binary crossentropy for multi-label classification and account for unknown classes
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])

    print("Network has been successfully compiled.")
    network.summary()

    # TODO check layer layout
    from tensorflow.keras.utils import plot_model
    plot_model(
        network,
        to_file= os.path.join(output_folder, "model.png"),
        show_shapes=True)

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
    network.save_weights(os.path.join(output_folder, "initial-random-weights.h5"))

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
        validation_shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
        train_index, val_index = next(validation_shuffle_split.split(X_train_and_val, y[train_and_val_index]))

        X_train, X_val = X_train_and_val[train_index], X_train_and_val[val_index]
        y_train, y_val = y_train_and_val[train_index], y_train_and_val[val_index]

        fold_report = f"Fold {fold}/{n_folds} (samples train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}"

        print(fold_report + ": starting the training process...")

        training_generator = ContigDataGenerator(X_train, y_train, **generator_params)
        validation_generator = ContigDataGenerator(X_val, y_val, **generator_params)
        testing_generator = ContigDataGenerator(X_test, y_test, **generator_params)
        network.load_weights(os.path.join(output_folder, "initial-random-weights.h5"))  # reset network to initial state

        train_history = network.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1,
            workers=1,  # TODO edit for multiprocessing
            use_multiprocessing=False
        )  # , callbacks=[early_stopping_callback])
        # see generator_params

        test_history = network.evaluate(
            testing_generator,
            epochs=epochs,
            verbose=1,
            workers=1,  # TODO edit for multiprocessing
            use_multiprocessing=False
        )

        print("Training process finished. Testing...")
        # TODO generators, evaluate
        # y_train_pred = network.predict(X_train).argmax(axis=1)
        # y_val_pred = network.predict(X_val).argmax(axis=1)
        # y_test_pred = network.predict(X_test).argmax(axis=1)

        # TODO use keras.metrics.CategoricalAccuracy for one-hot labels instead
        # from sklearn.metrics import accuracy_score
        # train_accuracy = accuracy_score(y_train, y_train_pred)
        # val_accuracy = accuracy_score(y_val, y_val_pred)
        # test_accuracy = accuracy_score(y_test, y_test_pred)

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
        network.save_weights(os.path.join(output_folder, f"fold-{fold}-weights.h5"))

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
    sys.exit(main())
