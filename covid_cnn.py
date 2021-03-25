
import h5py # this is used to save Keras models in the hdf5 format
import datetime
import os
import numpy as np
import pandas as pd
import sys

# imports for Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Dropout, Flatten, MaxPooling2D

# import for scikit-learn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def main() :

    # a few hard-coded values
    sequence_length = 31029 # length of a DNA/RNA sequence for the virus
    number_of_classes = 5 # TODO get directly from data
    batch_size = 50
    epochs = 1000 #500
    n_folds = 10
    starting_fold = 0 # this is used to stop and restart the testing on a sequence of folds (only works with a fixed random state)
    random_state = 42 # TODO change to None? for pseudo-random number generation initialized with time
    output_folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-keras-cnn-output"

    # create folder
    if not os.path.exists(output_folder) : os.makedirs(output_folder)

    # data pre-processing
    # prepare data in the correct format
    print("Reading data...")
    X_original = pd.read_csv("../NCBI_1503/data.csv", header=None).values
    y_original = pd.read_csv("../NCBI_1503/labels.csv", header=None).values
    number_of_classes = len(np.unique(y_original))
    print("X_original has shape:", X_original.shape)
    print("y_original has shape: %s, with %d unique classes, %s" % (str(y_original.shape), len(np.unique(y_original)), str(np.unique(y_original))))

    # also, normalize TODO move to cross-validation? Nah, each feature of the data has only 4 values
    # let's NOT normalize
    #scaler_x = StandardScaler()
    #X_original = scaler_x.fit_transform(X_original)

    # reshape input tensor to four dimensions
    X = np.reshape(X_original, (X_original.shape[0], 1, X_original.shape[1], 1))
    print("X, reshaped, has shape:", X.shape)

    # also reshape labels to one-hot encoding
    one_hot_encoder = OneHotEncoder()
    y = one_hot_encoder.fit_transform(y_original)
    print("y, reshaped to one-hot encoding, has shape:", y.shape)

    # TODO check if order in layer 1 is: conv+bias+relu+maxpool

    # shapes of the tensors in the network
    # (?, 1, 31029, 1)
    # (?, 1, 210, 130)
    # (?, 1, 1, 204)
    # (?, 1, 1, 150)
    # (?, 5)

    # input
    # python3 gen3.py 500 130 204 150 196 148 236 81 9 106 121 0 0 0

    # default values for the network
    units_1 = 12 #130
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

    # the loss is adjusted with a parameter for l2 regularization in each of the convolutional layers, called "beta" in the gen3.py script
    penalty_regularization_w_conv = 0.001
   
    # create the network
    print("Creating network...")
    network = Sequential()

    # 1 Layer
    network.add(Conv2D(units_1, input_shape=(1, sequence_length, 1), activation='relu', kernel_size=(1, conv_window_length_1), padding='same', use_bias=True, bias_initializer=keras.initializers.Constant(bias_vector_init_value), kernel_regularizer=keras.regularizers.l2(penalty_regularization_w_conv)))
    network.add(MaxPooling2D(pool_size=(1, pool_size_1), padding='same'))

    # 2 Layer
    #network.add(Conv2D(units_2, activation='relu', kernel_size=(1, conv_window_length_2), padding='same', use_bias=True, bias_initializer=keras.initializers.Constant(bias_vector_init_value), kernel_regularizer=keras.regularizers.l2(penalty_regularization_w_conv)))
    #network.add(MaxPooling2D(pool_size=(1, pool_size_2), padding='same'))

    # 3 Layer
    #network.add(Conv2D(units_3, activation='relu', kernel_size=(1, conv_window_length_3), padding='same', use_bias=True, bias_initializer=keras.initializers.Constant(bias_vector_init_value), kernel_regularizer=keras.regularizers.l2(penalty_regularization_w_conv)))
    #network.add(MaxPooling2D(pool_size=(1, pool_size_3), padding='same'))

    # Rectifier Layer, with dropout
    network.add(Flatten()) # convert tensor in N dimensions to tensor in 2
    network.add(Dense(units_4, activation='relu', use_bias=True, bias_initializer=keras.initializers.Constant(bias_vector_init_value)))
    network.add(Dropout(dropout_probability))

    # (Output) Softmax
    network.add(Dense(number_of_classes, activation='softmax', use_bias=True, bias_initializer=keras.initializers.Constant(bias_vector_init_value))) # there are five classes in the problem

    # instantiate the optimizer, with its learning rate
    from keras.optimizers import Adam
    optimizer = Adam(learning_rate=1e-5)
   
    # the loss is categorical crossentropy, as this is a classification problem
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Network has been successfully compiled.")
    network.summary()

    # a 'callback' in Keras is a condition that is monitored during the training process
    # here we instantiate a callback for an early stop, that is used to avoid overfitting
    from keras.callbacks.callbacks import EarlyStopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=100, verbose=1, restore_best_weights=False)

    # before going into the cross-validation, we actually save the initial random weights of the network, to reset it later before each fold
    network.save_weights(os.path.join(output_folder, "initial-random-weights.h5"))

    #  cross-validation
    from sklearn.model_selection import StratifiedShuffleSplit
    stratified_shuffle_split = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.1, random_state=random_state)
    fold = 0

    for train_and_val_index, test_index in stratified_shuffle_split.split(X, y_original) : # using y_original because this method needs numeric labels

        # skip folds until 'starting_fold', used to stop and restart evaluations
        if fold >= starting_fold :

            # stratified k-fold only splits the data in two, so training and validation are together
            X_train_and_val, X_test = X[train_and_val_index], X[test_index]
            y_train_and_val, y_test = y[train_and_val_index], y[test_index]

            # get test and validation set indexes, using a StratifiedShuffleSplit with just one split
            validation_shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
            train_index, val_index = next(validation_shuffle_split.split(X_train_and_val, y_original[train_and_val_index]))
           
            X_train, X_val = X_train_and_val[train_index], X_train_and_val[val_index]
            y_train, y_val = y_train_and_val[train_index], y_train_and_val[val_index]

            print("Fold %d/%d (samples train=%d, validation=%d, test=%d): starting the training process..." % (fold, n_folds, len(X_train), len(X_val), len(X_test)))
            network.load_weights(os.path.join(output_folder, "initial-random-weights.h5")) # reset network to initial state
            network.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val), shuffle=True)#, callbacks=[early_stopping_callback])

            print("Training process finished. Testing...")
            y_train_pred = network.predict(X_train)
            y_val_pred = network.predict(X_val)
            y_test_pred = network.predict(X_test)

            # in order to evaluate the network, we need to go back to a standard encoding
            y_train_labels = one_hot_encoder.inverse_transform(y_train)
            y_train_pred_labels = one_hot_encoder.inverse_transform(y_train_pred)
            y_val_labels = one_hot_encoder.inverse_transform(y_val)
            y_val_pred_labels = one_hot_encoder.inverse_transform(y_val_pred)
            y_test_labels = one_hot_encoder.inverse_transform(y_test)
            y_test_pred_labels = one_hot_encoder.inverse_transform(y_test_pred)

            from sklearn.metrics import accuracy_score
            train_accuracy = accuracy_score(y_train_labels, y_train_pred_labels)
            val_accuracy = accuracy_score(y_val_labels, y_val_pred_labels)
            test_accuracy = accuracy_score(y_test_labels, y_test_pred_labels)

            from sklearn.metrics import balanced_accuracy_score
            test_balanced_accuracy = balanced_accuracy_score(y_test_labels, y_test_pred_labels)

            print("Accuracy on training: %.4f, validation: %.4f, test: %.4f" % (train_accuracy, val_accuracy, test_accuracy))

            # save everything to a folder: fold; predictions on training, test, validation; model
            print("Saving information for the current fold...")

            # save model (divided in two parts: network layout and weights)
            network_json = network.to_json()
            with open(os.path.join(output_folder, "fold-%d-model.json" % fold), "w") as fp : fp.write(network_json)
            network.save_weights(os.path.join(output_folder, "fold-%d-weights.h5" % fold))

            # save information about the fold
            with open(os.path.join(output_folder, "global-summary.txt"), "a") as fp :
                fp.write("Fold %d/%d (samples train=%d, validation=%d, test=%d)\n" % (fold, n_folds, len(X_train), len(X_val), len(X_test)))
                fp.write("Accuracy on training: %.4f, validation: %.4f, test: %.4f\n" % (train_accuracy, val_accuracy, test_accuracy))

            # save data of the fold
            #x_column_names = ["feature_%d" % f for f in range(0, sequence_length) ]

            df_train = pd.DataFrame()
            df_train["y_true"] = y_train_labels.reshape(-1)
            df_train["y_pred"] = y_train_pred_labels.reshape(-1)
            #for i, c in enumerate(x_column_names) : df_train[c] = X_train[:,0,i,0].reshape(-1)
            df_train.to_csv(os.path.join(output_folder, "fold-%d-training.csv" % fold), index=False)

            df_val = pd.DataFrame()
            df_val["y_true"] = y_val_labels.reshape(-1)
            df_val["y_pred"] = y_val_pred_labels.reshape(-1)
            df_val.to_csv(os.path.join(output_folder, "fold-%d-validation.csv" % fold), index=False)

            df_test = pd.DataFrame()
            df_test["y_true"] = y_test_labels.reshape(-1)
            df_test["y_pred"] = y_test_pred_labels.reshape(-1)
            df_test.to_csv(os.path.join(output_folder, "fold-%d-test.csv" % fold), index=False)

        fold += 1

    return

if __name__ == "__main__" :
    sys.exit( main() )