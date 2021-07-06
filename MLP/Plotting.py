import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from MLP.Utils import argmin_index

def end_plotting():
    plt.show()

def plot_compare_outputs(train_output, watch_output, name: str, file_name: str = ''):
    #plt.figure()
    if watch_output is not None:
        fig, ax = plt.subplots()

        tupleify = lambda x: (x[0], x[1])

        ax.add_collection(LineCollection([[tupleify(train_output[i]), tupleify(watch_output[i])] for i in range(len(train_output))], linewidths=[0.2 for _ in range(len(train_output))]))

        plt.title('Scatter outputs with output-target lines: ' + name)
        plt.scatter(train_output[:,0], train_output[:,1], marker='o', s=1, color='blue', label='Model outputs')
        plt.scatter(watch_output[:,0], watch_output[:,1], marker='o', s=1, color='green', label='True Data')
        plt.legend()
        plt.draw()
        if file_name != '':
            plt.savefig(file_name)
    else:
        plt.figure()
        plt.title('Blind test outputs: ' + name)
        plt.scatter(train_output[:,0], train_output[:,1], marker='o', s=1, color='green', label='CUP outputs')
        plt.draw()
        if file_name != '':
            plt.savefig(file_name)
        

def plot_final_training_with_test_error(train_errors, watch_errors, name: str = 'MSE', file_name: str = '', skip_first_elements=0):
    plt.figure()

    plt.plot(train_errors, color='blue', label='Train error')
    if watch_errors != []:
        plt.plot(watch_errors, color='green', label='Test error', linestyle='dashdot')
    plt.legend()
    plt.title('Final training: ' + name)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)

def plot_final_training_with_test_accuracies(train_accuracies, test_accuracies=[], file_name: str = ''):
    plt.figure()
    plt.plot(train_accuracies, color='blue', label='Train accuracy')
    if test_accuracies != []:
        plt.plot(test_accuracies, color='green', label='Test accuracy', linestyle='dashdot')
    plt.legend()
    plt.title('Final Training: Classification accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)

def plot_weights_norms(weights_norms, title: str, file_name: str = ''):
    plt.figure()
    plt.plot(weights_norms, label='Weights norm')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)

def plot_gradient_norms(gradient_norms, title: str, file_name: str = ''):
    plt.figure()
    plt.plot(gradient_norms, label='Gradient norm')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    if file_name != '':
        plt.savefig(file_name)

def find_best(plots):
    return argmin_index(lambda t: t['best_val_error'], plots)

def plot_model(train, vals, label1, label2, *args, **kwargs):
    if train != []:
        plt.plot(train, label=label1, color='blue', *args, **kwargs)
    if vals != []:
        plt.plot(vals, label=label2, color='orange', linestyle='dashed', *args, **kwargs)

ALPHA = 0.7

def plot_model_selection_learning_curves(plots, metric=False, highlight_best=True, name='MSE', file_name=''):
    plt.figure()
    best_i = find_best(plots) if highlight_best else -1

    for i, results in enumerate(plots):
        plot_model(results['train_errors'] if not metric else results['metric_train_errors'],
                   results['val_errors'] if not metric else results['metric_val_errors'],
                   'Training error' if i == best_i else '',
                   'Validation error' if i == best_i else '',
                   alpha=1.0 if i == best_i else ALPHA)

    plt.legend()
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)

def plot_model_selection_accuracies(plots, highlight_best=True, name='Accuracy', file_name=''):
    plt.figure()

    best_i = find_best(plots) if highlight_best else -1

    for i, results in enumerate(plots):
        plot_model(results['train_accuracies'], results['val_accuracies'],
                   'Train accuracy' if i == best_i else '',
                   'Validation accuracy' if i == best_i else '',
                   alpha=1.0 if i == best_i else ALPHA)

    plt.legend()
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)
