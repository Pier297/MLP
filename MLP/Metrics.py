import matplotlib.pyplot as plt
import numpy as np

# TODO: Remove me, duplicate of MLP.LossFuncions.MSE
def MSE(model, X, Y, L2 = 0) -> float:
    e = 0.0
    for i in range(X.shape[0]):
        e += (Y[i] - model.predict(X[i]))**2
    squared_weights_sum = 0.0
    for layer in model.layers:
        for row_weights in layer.W:
            squared_weights_sum += np.linalg.norm(row_weights) ** 2
    return (e / (2 * X.shape[0]))[0][0] + L2 * squared_weights_sum


def accuracy(model, X, Y) -> float:
    corrects = 0
    for i in range(X.shape[0]):
        # TODO: the threshold and output values should be derived from somewhere
        if Y[i] == (1 if model.predict(X[i]) >= 0.5 else 0):
            corrects += 1
    return corrects / X.shape[0]


def plot_learning_curves(train_errors, val_errors=[], show=False):
    plt.figure()
    plt.plot(train_errors, label='Train error')
    if val_errors != []:
        plt.plot(val_errors, label='Validation error')
    plt.legend()
    plt.title('MSE')
    plt.xlabel('iteration')
    plt.ylabel('errors')
    plt.draw()
    if show:
        plt.show()


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