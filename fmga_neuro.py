'''
fmga to train a neural network to classify a spiral dataset
Author: Ameya Daigavane
Reference: http://cs231n.github.io/neural-networks-case-study/
Most of the code below is from the reference above.
'''

import fmga
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    reg = 1e-5 # regularization lambda
    h = 30  # size of hidden layer
    num_examples = X.shape[0]

    for j in range(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.0,1,N) # radius
      t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
      X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
      y[ix] = j
    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()

    def cross_entropy_loss(*args):
        # cast to numpy array
        np_args = np.asarray(args)

        # unpack and reshape
        W = np_args[:D * h].reshape(D, h)
        b = np_args[D * h: D * h + h].reshape(1, h)
        W2 = np_args[D * h + h: D * h + h + h * K].reshape(h, K)
        b2 = np_args[D * h + h + h * K:].reshape(1, K)

        # hidden layer with ReLU activation
        hidden_layer = np.maximum(0, np.dot(X, W) + b)  # note, ReLU activation

        # scores!
        scores = np.dot(hidden_layer, W2) + b2

        # get unnormalized probabilities
        exp_scores = np.exp(scores)

        # normalize them for each example
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # for the correct classes
        correct_logprobs = -np.log(probs[range(num_examples), y])

        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)

        return -(data_loss + reg_loss)

    bounds = [(-7, 7) for _ in range(D*h + h + h*K + K)]
    best_params = fmga.maximize(cross_entropy_loss, dimensions=(D*h + h + h*K + K), population_size=500, iterations=20,
                                boundaries=bounds, mutation_range=1, mutation_probability=0.15, elite_fraction=0.15)
    print(best_params.fitness)

    W = best_params.coordinates[:D*h].reshape(D, h)
    b = best_params.coordinates[D*h: D*h + h].reshape(1, h)
    W2 = best_params.coordinates[D*h + h: D*h + h + h*K].reshape(h, K)
    b2 = best_params.coordinates[D*h + h + h*K:].reshape(1, K)

    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print(np.mean(predicted_class == y))

    # plot the resulting classifier
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    # fig.savefig('fmga_spiral_neuro.png')