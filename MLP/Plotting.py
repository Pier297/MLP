import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from MLP.Utils import argmin_index

# Plotting.py

# Main general plotting procedures for both Monk and CUP files.

def end_plotting():
    plt.show()

def plot_compare_outputs(train_output, watch_output, name: str, file_name: str = ''):
    """
    Compare the outputs predicted by the model and the target ones
    as a scatter plot. This is useful for bidimensional outputs, such as the ML-CUP.
    The values can be optionally connected with a line expressing their euclidean distance.
    The watch_output can be omitted, and only the train will be shown.
    :param train_output: output of the model
    :param watch_output: expected output points
    :param name: graph title
    :param file_name: optional filename to save the graph on
    """

    if watch_output is not None:
        fig, ax = plt.subplots()

        tupleify = lambda x: (x[0], x[1])

        # Construct the lines joining target and output points

        ax.add_collection(LineCollection([[tupleify(train_output[i]), tupleify(watch_output[i])] for i in range(len(train_output))],\
                                          linewidths=[0.2 for _ in range(len(train_output))]))

        plt.title('Scatter outputs with output-target lines: ' + name)

        # Plot both datasets with different colors

        plt.scatter(train_output[:,0], train_output[:,1], marker='o', s=1, color='blue', label='Model outputs')
        plt.scatter(watch_output[:,0], watch_output[:,1], marker='o', s=1, color='green', label='True Data')
        plt.legend()
        plt.draw()
        if file_name != '':
            plt.savefig(file_name)
    else:
        # If no target dataset is provided, simply output them as they are on the graph
        # (this is useful for blind datasets which have no output to compare on, while still
        # providing a visual comparison)

        plt.figure()
        plt.title('Blind test outputs: ' + name)
        plt.scatter(train_output[:,0], train_output[:,1], marker='o', s=1, color='green', label=name)
        plt.draw()
        if file_name != '':
            plt.savefig(file_name)

def plot_final_training_with_test_error(train_errors, watch_errors, title: str = 'MSE', name='Test', file_name: str = '', skip_first_elements=0):
    """
    Plot the given two learning curves, considered as training and watch output.
    :param train_errors: error graph of the model
    :param watch_errors: secondary watch error graph to run parallel
    :param title: graph title
    :param file_name: optional filename to save the graph on
    """
    plt.figure()

    plt.plot(train_errors, color='blue', label='Train error')
    if watch_errors != []:
        plt.plot(watch_errors, color='green', label=f'{name} accuracy', linestyle='dashdot')
    plt.legend()
    plt.title('Final training: ' + title)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)

def plot_final_training_with_test_accuracies(train_accuracies, test_accuracies=[], name='Test', file_name: str = ''):
    """
    Plot the given two accuracy performance curves, considered as training and watch output.
    :param train_accuracies: accuracy graph of the model
    :param test_accuracies: secondary watch error graph to run parallel
    :param name: graph title
    :param file_name: optional filename to save the graph on
    """
    plt.figure()
    plt.plot(train_accuracies, color='blue', label='Train accuracy')
    if test_accuracies != []:
        plt.plot(test_accuracies, color='green', label=f'{name} accuracy', linestyle='dashdot')
    plt.legend()
    plt.title('Final Training: Classification accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)

def plot_model_selection_learning_curves(plots, metric=False, highlight_best=True, name='MSE', file_name='', alpha = 0.7):
    """
    Plot the given validation plots given by the cross validation function.
    This function can display indifferently both trials and folds.
    :param plots: list of validation results
    :param highlight_best: produce a different opacity for the best performing fold/trial
    :param name: graph title
    :param file_name: optional filename to save the graph on
    :param alpha: optional transparency value of the graphs
    """

    # Find the best model in the trials so that it can be displayed with a lesser transparency

    plt.figure()
    best_i = find_best(plots) if highlight_best else -1

    # For each of the given plots, call the generic plot_model function

    for i, results in enumerate(plots):
        plot_model(results['train_errors'] if not metric else results['metric_train_errors'],
                   results['val_errors'] if not metric else results['metric_val_errors'],
                   'Training error' if i == best_i else '',
                   'Validation error' if i == best_i else '',
                   alpha=1.0 if i == best_i else alpha)

    plt.legend()
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)

def plot_model_selection_accuracies(plots, highlight_best=True, name='Accuracy', file_name='', alpha = 0.7):
    """
    Plot the given two accuracy performance plots, considered as training and validation.
    :param plots: list of validation results
    :param name: graph title
    :param file_name: optional filename to save the graph on
    :param alpha: optional transparency value of the graphs
    """

    # Find the best model in the trials so that it can be displayed with a lesser transparency

    plt.figure()
    best_i = find_best(plots) if highlight_best else -1

    # For each of the given plots, call the generic plot_model function

    for i, results in enumerate(plots):
        plot_model(results['train_accuracies'], results['val_accuracies'],
                   'Train accuracy' if i == best_i else '',
                   'Validation accuracy' if i == best_i else '',
                   alpha=1.0 if i == best_i else alpha)

    plt.legend()
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)

def plot_model(train, vals, label1, label2, *args, **kwargs):
    """
    Plot the given two validation and training graphs combined.
    This is a simple wrapper on the plt.plot function, which is encapsulated with this helper function
    to constistenly distinguish validation and training.
    :param train: training errors vector
    :param vals:  vals errors vector
    :param label1: optional label for training
    :param label2: optional label for validation
    """
    if train != []:
        plt.plot(train, label=label1, color='blue', *args, **kwargs)
    if vals != []:
        plt.plot(vals, label=label2, color='orange', linestyle='dashed', *args, **kwargs)

def find_best(plots):
    """Helper function to identify the plot with best validation error."""
    return argmin_index(lambda t: t['best_val_error'], plots)
