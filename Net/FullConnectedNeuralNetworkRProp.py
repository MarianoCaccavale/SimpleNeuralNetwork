# Utilizzo la Resilient BackPropagation di tipo "without weight-backtracking"
from typing import List, Callable
import numpy as np
from Net.Layer import *
from Net.FullConnectedNeuralNetwork import FullConnectedNeuralNetwork


class FullConnectedNeuralNetworkRprop(FullConnectedNeuralNetwork):

  def __init__(self, layers: List[Layer], cost_function: Callable[[np.array, np.array], float ],
               rprop_eta_plus: float = 1.2, rprop_eta_minus: float = 0.5,
               max_step_size: float = 50., min_step_size:float = 1e-6):

    super().__init__(layers, cost_function, 1)

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
      self.step_sizes.append(np.full(layer.weights.shape, self.max_step_size/2))

  def _compute_gradient_change(self, last_gradient: np.array, current_gradient: np.array):
    return np.multiply(np.sign(last_gradient), np.sign(current_gradient))

  def _update_step_size(self, layer_index: int, gradient_change: np.ndarray):

    self.step_sizes[layer_index] = np.where(gradient_change > 0, 
                                            np.minimum(self.step_sizes[layer_index] * self.rprop_eta_plus, self.max_step_size), 
                                            self.step_sizes[layer_index])
  
    self.step_sizes[layer_index] = np.where(gradient_change < 0, 
                                            np.maximum(self.step_sizes[layer_index] * self.rprop_eta_minus, self.min_step_size), 
                                            self.step_sizes[layer_index])

    
  # def _improve_gradient(self, gradient_diffs: np.array, current_gradient: np.array):
  #   for row_index in range(0, gradient_diffs.shape[0]):
  #     for col_index in range(0, gradient_diffs.shape[1]):
  #       if gradient_diffs is not None and gradient_diffs[row_index, col_index] < 0:
  #         current_gradient[row_index][col_index] = 0
  #   return current_gradient

  def _back_propagate(self, ground_truth: np.array):

    last_dCdA = 0
    layer_index = 0

    weights_delta = []
    biases_delta = []

    gradient_change = None

    for layer_index in range(len(self.layers)-1, -1, -1):

      layer = self.layers[layer_index]

      # Output layer
      if layer_index == len(self.layers) - 1:
        dCdA = self.cost_function.compute_derivate(layer.A, ground_truth)[:, np.newaxis]
        dZdW = self.layers[layer_index - 1].A[:, np.newaxis].T
        dCdW = np.dot(dCdA, dZdW) / layer.number_of_neurons

      else:
        dAdZ = layer.activation_function_derivative[:, np.newaxis]
        first_part = np.dot(self.layers[layer_index + 1].weights.T, last_dCdA)
        dCdA = first_part *  dAdZ
        dCdW = np.dot(dCdA, layer.X[:, np.newaxis].T) / layer.number_of_neurons

      # Se invece ho già salvato la derivata della funzione di costo aggiorno gli step moltiplicando l'attuale costo 
      # per il costo del "tempo" precedente
      if self.last_gradient_per_layer[layer_index] is not None:
        gradient_change = self._compute_gradient_change(self.last_gradient_per_layer[layer_index], dCdW)
        self._update_step_size(layer_index, gradient_change)

      dw = np.multiply(-np.sign(dCdW), self.step_sizes[layer_index])

      #if gradient_change is not None:
      #  dCdW = self._improve_gradient(gradient_change, dCdW)

      db = np.sum(dCdA, axis = 1, keepdims=True) / layer.number_of_neurons

      self.last_gradient_per_layer[layer_index] = dCdW

      weights_delta.append(dw)
      biases_delta.append(db)

      last_dCdA = dCdA

    weights_delta.reverse()
    biases_delta.reverse()

    return weights_delta, biases_delta