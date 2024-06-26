# Utilizzo la Resilient BackPropagation di tipo "without weight-backtracking"
from typing import List, Callable
import numpy as np
from Net.Layer import *
from Net.FullConnectedNeuralNetwork import FullConnectedNeuralNetwork


class FullConnectedNeuralNetworkRprop(FullConnectedNeuralNetwork):

  def __init__(self, layers: List[Layer], cost_function: Callable[[np.ndarray, np.ndarray], float ],
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

    self.last_weight_gradient_per_layer = [None] * len(layers)
    self.last_bias_gradient_per_layer = [None] * len(layers)

  def _initialize_step_sizes(self):
    self.weight_step_sizes = []
    self.bias_step_sizes = []

    for layer in self.layers:
      self.weight_step_sizes.append(np.full(layer.weights.shape, 0.005))
      self.bias_step_sizes.append(np.full(layer.biases.shape, 0.005))

  def _compute_gradient_change(self, last_gradient: np.ndarray, current_gradient: np.ndarray):
    return np.multiply(np.sign(last_gradient), np.sign(current_gradient))

  def _update_step_size(self, layer_index: int, weight_gradient_change: np.ndarray, bias_gradient_change: np.ndarray):

    self.weight_step_sizes[layer_index] = np.where(weight_gradient_change > 0, 
                                            np.minimum(self.weight_step_sizes[layer_index] * self.rprop_eta_plus, self.max_step_size), 
                                            self.weight_step_sizes[layer_index])
  
    self.weight_step_sizes[layer_index] = np.where(weight_gradient_change < 0, 
                                            np.maximum(self.weight_step_sizes[layer_index] * self.rprop_eta_minus, self.min_step_size), 
                                            self.bias_step_sizes[layer_index])
    
    self.bias_step_sizes[layer_index] = np.where(bias_gradient_change > 0, 
                                            np.minimum(self.bias_step_sizes[layer_index] * self.rprop_eta_plus, self.max_step_size), 
                                            self.bias_step_sizes[layer_index])
  
    self.bias_step_sizes[layer_index] = np.where(bias_gradient_change < 0, 
                                            np.maximum(self.bias_step_sizes[layer_index] * self.rprop_eta_minus, self.min_step_size), 
                                            self.bias_step_sizes[layer_index])

  def _improve_gradient(self, gradient_diffs: np.ndarray, current_gradient: np.ndarray):
    return np.where(gradient_diffs >= 0, current_gradient, 0)

  def _update_parameters(self, dw: List[np.ndarray], db: List[np.ndarray]):
    for layer_index in range(len(self.layers)):

      if self.last_weight_gradient_per_layer[layer_index] is not None:
        weight_gradient_change = self._compute_gradient_change(self.last_weight_gradient_per_layer[layer_index], dw[layer_index])
        bias_gradient_change = self._compute_gradient_change(self.last_bias_gradient_per_layer[layer_index], db[layer_index])
        self._update_step_size(layer_index, weight_gradient_change, bias_gradient_change)
        # dw[layer_index] = self._improve_gradient(gradient_change, dw[layer_index])

      self.last_weight_gradient_per_layer[layer_index] = dw[layer_index]
      self.last_bias_gradient_per_layer[layer_index] = db[layer_index]

      delta_weight = (-np.sign(dw[layer_index]) * self.weight_step_sizes[layer_index])
      delta_bias = -(-np.sign(db[layer_index]) * self.bias_step_sizes[layer_index])
      self.layers[layer_index].update_parameters(delta_weight, delta_bias)