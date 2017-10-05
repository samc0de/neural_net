# Implements 2 class classifier from wildml example.
import numpy as np
import sklearn
from matplotlib import pyplot as plt

# Seed random module for deterministic behaviour.
np.random.seed(0)


def generate_dataset(shape='moons', points=200, noise=0.2, show_plot=False):
  """Generates a random dataset for classification."""
  if shape != 'moons':
    raise NotImplementedError('Only moon shape is implemented ATM.')

  # Update the default values of this function if needed.
  X, y = sklearn.datasets.make_moons(n_samples=points, noise=noise)

  if show_plot:
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

  return X


def main(args):
  # TODO: Read below params (& # hidden layers etc) from command line
  # arguments.
  inputs = generate_dataset(points=200, noise=0.2)  # these too from cmd args.
  num_examples = len(inputs) # training set size
  nn_input_dim = 2 # input layer dimensionality
  nn_output_dim = 2 # output layer dimensionality

  # Gradient descent parameters (I picked these by hand)
  epsilon = 0.01 # learning rate for gradient descent
  reg_lambda = 0.01 # regularization strength
