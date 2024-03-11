# Utilizzo la Resilient BackPropagation di tipo "without weight-backtracking"
from typing import List, Callable
import numpy as np
from Net.Layer import *
from Net.FullConnectedNeuralNetwork import FullConnectedNeuralNetwork


class FullConnectedNeuralNetworkRprop(FullConnectedNeuralNetwork):

  def __init__(self, layers: List[Layer], cost_function: Callable[[np.array, np.array], float ],
               rprop_eta_plus: float = 1.2, rprop_eta_minus: float = 0.5,
               max_step_size: float = 10., min_step_size:float = 1e-6):

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
      self.step_sizes.append(np.full(layer.weights.shape, 0.01))

  def _compute_gradient_change(self, last_gradient: np.array, current_gradient: np.array):
    return np.multiply(np.sign(last_gradient), np.sign(current_gradient))

  def _update_step_size(self, layer_index: int, gradient_change: np.ndarray):

    self.step_sizes[layer_index] = np.where(gradient_change > 0, 
                                            np.minimum(self.step_sizes[layer_index] * self.rprop_eta_plus, self.max_step_size), 
                                            self.step_sizes[layer_index])
  
    self.step_sizes[layer_index] = np.where(gradient_change < 0, 
                                            np.maximum(self.step_sizes[layer_index] * self.rprop_eta_minus, self.min_step_size), 
                                            self.step_sizes[layer_index])

  def _back_propagate(self, ground_truth: np.array):

    last_delta_valore = 0
    layer_index = 0

    weights_delta = []
    biases_delta = []

    for layer_index in range(len(self.layers)-1, -1, -1):

      layer = self.layers[layer_index]
      
      dAdZ = layer.activation_function_derivative[:, np.newaxis] # 10, 1
      dZdW = self.layers[layer_index - 1].A[:, np.newaxis] if layer_index != 0 else self.layers[layer_index].X[:, np.newaxis] # 256, 1

      # Output layer
      if layer_index == len(self.layers) - 1:
        dCdA = self.cost_function.compute_derivate(layer.A, ground_truth)[:, np.newaxis] # 10, 1
      else:
        dCdA = np.dot(self.layers[layer_index + 1].weights.T, last_delta_valore)

      delta_valore = np.multiply(dAdZ, dCdA) #10, 1
      dw = np.dot(delta_valore, dZdW.T)# / layer.number_of_neurons
      db = np.sum(dCdA, axis = 1, keepdims=True)# / layer.number_of_neurons

      if self.last_gradient_per_layer[layer_index] is not None:
        gradient_change = self._compute_gradient_change(self.last_gradient_per_layer[layer_index], dw)
        self._update_step_size(layer_index, gradient_change)

      weights_delta.append(-(-np.sign(dw) * self.step_sizes[layer_index]))
      biases_delta.append(db)

      last_delta_valore = delta_valore
      self.last_gradient_per_layer[layer_index] = dw

    # I delta dei pesi e i delta dei biases sono "al contrario", ovvero quelli in posizione 0 della lista in realtà sono per il layer di output, quindi faccio il
    # reverse della lista per averli sistemati
    weights_delta.reverse()
    biases_delta.reverse()

    return weights_delta, biases_delta