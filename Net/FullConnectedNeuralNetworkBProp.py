from numpy import ndarray
from Net.FullConnectedNeuralNetwork import *

class FullConnectedNeuralNetworkBprop(FullConnectedNeuralNetwork):
  def _update_parameters(self, dw: List[np.ndarray], db: List[np.ndarray]):
    for layer_index in range(len(self.layers)):
      delta_weight = -(dw[layer_index] * self.learning_rate)
      delta_bias = -(db[layer_index] * self.learning_rate)
      self.layers[layer_index].update_parameters(delta_weight, delta_bias)
