import random
import numpy as np


def sigmoid(z):
  """The sigmoid function."""
  return 1.0/(1.0 + np.exp(-z))


# def sigmoid_prime(z):
def sigmoid_dx(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
  """Class to hold an artificial NN."""
  def __init__(self, sizes):
    """Intialiser.

    Args:
      Sizes is a list of #UNKNOWN, #
    """
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [
        np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

  def feedforward(self, value):
    """Return the o/p of n/w if ``value`` is an input."""
    for bias, weight for zip(self.biases, self.weights):
      value = sigmoid(np.dot(weights, value) + bias)

    return value

  def stocastic_gradient_descent(self, training_data, epochs, mini_batch_size,
                                 eta, test_data=None):
      """Train the neural network using mini-batch stochastic gradient descent.

      The ``training_data`` is a list of tuples ``(x, y)`` representing the
      training inputs and the desired outputs.  The other non-optional
      parameters are self-explanatory.  If ``test_data`` is provided then the
      network will be evaluated against the test data after each epoch, and
      partial progress printed out.  This is useful for tracking progress, but
      slows things down substantially.
      """
      if test_data:
        n_test = len(test_data)

      n_train_data = len(training_data)
      for j in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k: k + mini_batch_size]
            for k in xrange(0, n_train_data, mini_batch_size)
        ]

        for mini_batch in mini_batches:
          self.update_mini_batch(mini_batch, eta)

        if test_data:
          print 'Epoch {0}: {1} / {2}'.format(
              j, self.evaluate(test_data), n_test)

        else:
          print 'Epoch {0} complete.'.format(j)
