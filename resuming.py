
import os
import re
import warnings

import numpy as np
import tensorflow as tf


def max_regex(path, regex, min_idx=0):
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
        split = os.path.splitext(target_path)
        # keep the path prefix only if the target is a weights file:
        if split[1] == ".index" or split[1].startswith(".data"):
            target_path = split[0]
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
    starting_epoch = 0
    epoch_path = None
    fold_regex = re.compile("fold-(\\d*)")
    epoch_regex = re.compile("epoch-(\\d*).*")
    starting_fold, fold_path = max_regex(output_folder, fold_regex)
    if fold_path is not None:
        starting_epoch, epoch_path = max_regex(fold_path, epoch_regex)
        if epoch_path is not None:
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
