from math import ceil
import matplotlib.pyplot as plt

# adapted from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data


def plot_metrics(metrics: "list[str]", history):
    """Constructs a graph of selected metrics based on the training history.

    This function itself does not plot not 

    Parameters
    ----------
    history : History
        The History object returned by calling `model.fit`.
    metrics : list[str]
        The list of metric names to plot; the corresponding data will be
        plotted for both training and validation. The `model.metrics_names`
        property can be used to plot all values, as entries starting with
        `val_` will be ignored.
    """
    metrics = [name for name in metrics if not name.startswith("val_")]
    for idx, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(ceil(len(metrics) / 2), 2, idx + 1)
        plt.plot(history.epoch,
                 history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
