import matplotlib.pyplot as plt
from MLP.Utils import argmin

def plot_final_training_with_test_error(train_errors, test_errors, show=False, name: str = 'MSE', file_name: str = ''):
    plt.figure()
    plt.plot(train_errors, color='blue', label='Train error')
    if test_errors != []:
        plt.plot(test_errors, color='green', label='Test error')
    plt.legend()
    plt.title('Final training: ' + name)
    plt.xlabel('iteration')
    plt.ylabel('errors')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)
    if show:
        plt.show()

def plot_final_training_with_test_accuracies(train_accuracies, test_accuracies=[], show=False, file_name: str = ''):
    plt.figure()
    plt.plot(train_accuracies, color='blue', label='Train accuracy')
    if test_accuracies != []:
        plt.plot(test_accuracies, color='green', label='Test accuracy')
    plt.legend()
    plt.title('Final Training: Classification accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)
    if show:
        plt.show()

def plot_weights_norms(weights_norms, title: str, file_name: str = ''):
    plt.figure()
    plt.plot(weights_norms, label='Weights norm')
    plt.legend()
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('norm')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)

def plot_gradient_norms(gradient_norms, title: str, file_name: str = ''):
    plt.figure()
    plt.plot(gradient_norms, label='Gradient norm')
    plt.legend()
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('norm')
    if file_name != '':
        plt.savefig(file_name)

def find_best(trials):
    return argmin(lambda t: t['val_error'], trials)


def plot_model(train, vals, label1, label2, *args, **kwargs):
    if train != []:
        plt.plot(train, label=label1, color='blue', *args, **kwargs)
    if vals != []:
        plt.plot(vals, label=label2, color='orange', *args, **kwargs)

def plot_model_selection_learning_curves(trials, highlight_best=True, name='MSE', show=False, file_name=''):
    plt.figure()

    best_i = find_best(trials) if highlight_best else -1

    for i, results in enumerate(trials):
        plot_model(results['train_errors'], results['val_errors'],
                   'Training error' if i == best_i else '', 'Validation error' if i == best_i else '',
                   alpha=1.0 if i == best_i else 0.1)

    plt.legend()
    plt.title(name)
    plt.xlabel('iteration')
    plt.ylabel('errors')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)
    if show:
        plt.show()


def plot_model_selection_accuracies(trials, highlight_best=True, name='Accuracy', show=False, file_name=''):
    plt.figure()

    best_i = find_best(trials) if highlight_best else -1

    for i, results in enumerate(trials):
        plot_model(results['train_accuracies'], results['val_accuracies'],
                   'Train accuracy' if i == best_i else '', 'Validation accuracy' if i == best_i else '',
                   alpha=1.0 if i == best_i else 0.1)

    plt.legend()
    plt.title(name)
    plt.xlabel('iteration')
    plt.ylabel('accuracies')
    plt.draw()
    if file_name != '':
        plt.savefig(file_name)
    if show:
        plt.show()