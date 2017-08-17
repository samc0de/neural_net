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
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1)]
