from MLP.Regularizers import early_stopping
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(train_errors, val_errors=[], early_stopping_epoch=-1, show=False, name: str = 'MSE'):
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

def plot_accuracies(train_accuracies, val_accuracies=[], early_stopping_epoch=-1, show=False):
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
    plt.savefig('accuracies.png')