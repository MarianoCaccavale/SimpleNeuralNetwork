# Utilizzo la Resilient BackPropagation di tipo "without weight-backtracking"
from typing import List, Callable
import numpy as np
from Layer import *
from Net.FullConnectedNeuralNetwork import FullConnectedNeuralNetwork


class FullConnectedNeuralNetworkRprop(FullConnectedNeuralNetwork):

  def __init__(self, layers: List[Layer], cost_function: Callable[[np.array, np.array], float ], learning_rate: int = 0.01,
               rprop_eta_plus: float = 1.5, rprop_eta_minus: float = 0.5,
               max_step_size: float = 10., min_step_size:float = 0.):

    super().__init__(layers, cost_function, learning_rate)

    if (rprop_eta_minus >= 1 or rprop_eta_minus <= 0):
      raise Exception("Il parametro eta- della Resilient Backpropagation deve essere compreso tra ]0, 1[")

    if (rprop_eta_plus <= 1):
      raise Exception("Il parametro eta+ della Resilient Backpropagation deve essere compreso tra ]1, inf[")

    self.rprop_eta_plus = rprop_eta_plus
    self.rprop_eta_minus = rprop_eta_minus

    self.max_step_size = max_step_size
    self.min_step_size = min_step_size
    self._initialize_step_sizes()

    self.last_gradient_per_layer = [None] * len(layers)

  def _initialize_step_sizes(self):
    self.step_sizes = []

    for layer in self.layers:
      self.step_sizes.append(np.zeros(layer.weights.shape))
      # self.step_sizes.append(np.random.uniform(low = self.min_step_size, high = self.max_step_size, size = layer.weights.shape))

  def _compute_gradient_change(self, last_gradient: np.array, current_gradient: np.array):
    return np.multiply(np.sign(last_gradient), np.sign(current_gradient))

  def _update_step_size(self, layer_index: int, gradient_change: np.ndarray):
    for row_index in range(0, gradient_change.shape[0]):
      for col_index in range(0, gradient_change.shape[1]):
        if gradient_change[row_index][col_index] > 0:
          self.step_sizes[layer_index][row_index][col_index] = min(self.step_sizes[layer_index][row_index][col_index] * self.rprop_eta_plus, self.max_step_size)
        elif gradient_change[row_index][col_index] < 0:
          self.step_sizes[layer_index][row_index][col_index] = max(self.step_sizes[layer_index][row_index][col_index] * self.rprop_eta_minus, self.min_step_size)

  def _improve_gradient(self, gradient_diffs: np.array, current_gradient: np.array):
    for row_index in range(0, gradient_diffs.shape[0]):
      for col_index in range(0, gradient_diffs.shape[1]):
        if gradient_diffs is not None and gradient_diffs[row_index, col_index] < 0:
          current_gradient[row_index][col_index] = 0

    return current_gradient

  def _back_propagate(self, ground_truth: np.array):

    last_dCdA = 0
    layer_index = 0

    weights_delta = []
    biases_delta = []

    gradient_change = None

    for layer_index in reversed(range(0, len(self.layers))):

      layer = self.layers[layer_index]

      # Output layer
      if layer_index == len(self.layers) - 1:
        dCdA = self.cost_function.compute_derivate(layer.A, ground_truth)
        # Faccio diventare il vettore colonna dZ una matrice con una sola colonna, che è proprio il vettore colonna. Questo per far quadrare le dimensioni nelle moltiplicazioni
        # successive
        dCdA = dCdA[:, np.newaxis]
        dZdW = self.layers[layer_index - 1].A[:, np.newaxis].T
        dCdW = (1/layer.number_of_neurons) * np.dot(dCdA, dZdW)

        # Se invece ho già salvato la derivata della funzione di costo aggiorno gli step moltiplicando l'attuale costo per il costo del "tempo" precedente
        if self.last_gradient_per_layer[layer_index] is not None:
          gradient_change = self._compute_gradient_change(self.last_gradient_per_layer[layer_index], dCdW)
          self._update_step_size(layer_index, gradient_change)

      else:
        # N.B. dCdA[i]       = w[i+1].T x dZ[i+1] * g'[i](Z[i])
        #      dCdA[i]       = first_part         * dAdZ
        #      first_part    = np.dot(self.layers[layer_index + 1].weights.T, last_dCdA)
        #      dAdZ          = compute_derivate(layer.Z)

        dAdZ = layer.activation_function_derivative[:, np.newaxis]

        first_part = np.dot(self.layers[layer_index + 1].weights.T, last_dCdA)
        dCdA = first_part *  dAdZ
        dCdW = (1/layer.number_of_neurons) * np.dot(dCdA, layer.X[:, np.newaxis].T)

        # Se invece ho già salvato la derivata della funzione di costo aggiorno gli step moltiplicando l'attuale costo per il costo del "tempo" precedente
        if self.last_gradient_per_layer[layer_index] is not None:
          gradient_change = self._compute_gradient_change(self.last_gradient_per_layer[layer_index], dCdW)
          self._update_step_size(layer_index, gradient_change)

      dw = np.multiply(-np.sign(dCdW), self.step_sizes[layer_index])

      if gradient_change is not None:
        dCdW = self._improve_gradient(gradient_change, dCdW)

      db = (1/layer.number_of_neurons) * np.sum(dCdA, axis = 1, keepdims=True)

      self.last_gradient_per_layer[layer_index] = dCdW

      weights_delta.append(-dw)
      biases_delta.append(db)

      last_dCdA = dCdA

    weights_delta.reverse()
    biases_delta.reverse()

    return weights_delta, biases_delta