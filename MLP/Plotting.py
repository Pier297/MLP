import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(train_errors, val_errors=[], show=False, name: str = 'MSE'):
    plt.figure()
    plt.plot(train_errors, label='Train error')
    if val_errors != []:
        plt.plot(val_errors, label='Validation error')
    plt.legend()
    plt.title(name)
    plt.xlabel('iteration')
    plt.ylabel('errors')
    plt.draw()
    if show:
        plt.show()
    plt.savefig('learning_curves.png')

def plot_accuracies(train_accuracies, val_accuracies=[], show=False):
    plt.figure()
    plt.plot(train_accuracies, label='Train accuracy')
    if val_accuracies != []:
        plt.plot(val_accuracies, label='Validation accuracy')
    plt.legend()
    plt.title('Classification accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.draw()
    if show:
        plt.show()
    plt.savefig('accuracies.png')