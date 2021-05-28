
import os
import re
import warnings

import numpy as np
import tensorflow as tf


def max_regex(path, regex, min_idx=0):
    # XXX TODO
    m = min_idx - 1
    target_path = None
    for filename in os.listdir(path):
        match = regex.match(filename)
        if match:
            idx = int(match.group(1))
            if idx > m:
                m = idx
                target_path = filename
    if m < min_idx:
        # caught nothing, but respecting the contract:
        m = min_idx
    else:
        # return the full path
        target_path = os.path.join(path, target_path)
    return m, target_path


def resume(output_folder, n_epochs):
    """Returns information allowing to resume training.

    Parameters
    ----------
    output_folder : path-like
        Path-like to the folder were fold output is stored.
    n_epochs : int
        Maximum number of epochs trained per fold.

    Returns
    -------
    starting_fold, starting_epoch, checkpoint_path: int * int * path-like
        starting_fold and starting_epoch are the points you want to resume
        or start training at. You will need to train from scratch (new 
        fold) if and only if checkpoint_path is set to None, else a path
        to the checkpoint to load the model from will be given.
    """
    epoch_path = None
    fold_regex = re.compile("fold-(\\d*)")
    epoch_regex = re.compile("epoch-(\\d*).*")
    starting_fold, fold_path = max_regex(output_folder, fold_regex)
    if fold_path is not None:
        starting_epoch, epoch_path = max_regex(fold_path, epoch_regex)
        if epoch_path is not None:
            starting_epoch += 1  # don't start from the epoch we already have results for !
            if starting_epoch >= n_epochs:
                starting_fold += 1
                starting_epoch = 0
                epoch_path = None
    return starting_fold, starting_epoch, epoch_path

# mostly copied from `tf.keras.utils.tf_utils`:


def _sync_to_numpy_or_python_type(tensors):
    """Syncs and converts a structure of `Tensor`s to `NumPy` arrays or Python scalar types.

    For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
    it converts it to a Python type, such as a float or int, by calling
    `result.item()`.

    Numpy scalars are converted, as Python types are often more convenient to deal
    with. This is especially useful for bfloat16 Numpy scalars, which don't
    support as many operations as other Numpy values.

    Async strategies (such as `TPUStrategy` and `ParameterServerStrategy`) are
    forced to
    sync during this process.

    Args:
      tensors: A structure of tensors.

    Returns:
      `tensors`, but scalar tensors are converted to Python types and non-scalar
      tensors are converted to Numpy arrays.
    """
#   if isinstance(tensors, coordinator_lib.RemoteValue):
#     return tensors.fetch()

    def _to_single_numpy_or_python_type(t):
        if isinstance(t, tf.Tensor):
            x = t.numpy()
            return x.item() if np.ndim(x) == 0 else x
        return t  # Don't turn ragged or sparse tensors to NumPy.

    return tf.nest.map_structure(_to_single_numpy_or_python_type, tensors)


class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 save_traces=True,
                 mode='auto',
                 save_freq='epoch',
                 options=None,
                 **kwargs):
        super().__init__(filepath,
                         monitor=monitor,
                         verbose=verbose,
                         save_best_only=save_best_only,
                         save_weights_only=save_weights_only,
                         mode=mode,
                         save_freq=save_freq,
                         options=options,
                         **kwargs)
        self.save_traces = save_traces

    # mostly copied from `tf.keras.callbacks`:
    def _save_model(self, epoch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        # HACK
        tf.print("saving", output_stream="file://../tmp/file_list.txt")
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = _sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn(
                            f'Can save best model only with {self.monitor} available, skipping.')
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s' % (epoch + 1, self.monitor,
                                                               self.best, current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    filepath, overwrite=True, options=self._options)
                            else:
                                self.model.save(
                                    filepath, overwrite=True, save_traces=self.save_traces, options=self._options)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' %
                              (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(
                            filepath, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True,
                                        save_traces=self.save_traces,
                                        options=self._options)

                self._maybe_remove_file()
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing '
                                  'directory: {}'.format(filepath))
                # Re-throw the error for any other causes.
                raise e
