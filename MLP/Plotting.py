import matplotlib.pyplot as plt
import numpy as np
from MLP.experiments.utils import argmin

""" def plot_learning_curves2(train_errors, val_errors=[], early_stopping_epoch=-1, show=False, name: str = 'MSE'):
    plt.figure()
    plt.plot(train_errors, label='Train error')
    if val_errors != []:
        plt.plot(val_errors, label='Validation error')
        if early_stopping != -1:
            plt.plot(early_stopping_epoch, val_errors[early_stopping_epoch], 'go', label='Early stopping point')
    plt.legend()
    plt.title(name)
    plt.xlabel('iteration')
    plt.ylabel('errors')
    plt.draw()
    if show:
        plt.show()
    plt.savefig('learning_curves.png')

def plot_accuracies2(train_accuracies, val_accuracies=[], early_stopping_epoch=-1, show=False):
    plt.figure()
    plt.plot(train_accuracies, label='Train accuracy')
    if val_accuracies != []:
        plt.plot(val_accuracies, label='Validation accuracy')
        if early_stopping != -1:
            plt.plot(early_stopping_epoch, val_accuracies[early_stopping_epoch], 'go', label='Early stopping point')
    plt.legend()
    plt.title('Classification accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.draw()
    if show:
        plt.show()
    plt.savefig('accuracies.png') """


def find_best(trials):
    return argmin(lambda t: t['val_error'], trials)    


def plot_model(train, vals, label1, label2, *args, **kwargs):
    if train != []:
        plt.plot(train, label=label1, color='blue', *args, **kwargs)
    if vals != []:
        plt.plot(vals, label=label2, color='orange', *args, **kwargs)

def plot_learning_curves(trials, highlight_best=True, name='MSE', show=False, file_name=''):
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


def plot_accuracies(trials, highlight_best=True, name='Accuracy', show=True, file_name=''):
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