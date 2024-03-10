from Net.FullConnectedNeuralNetwork import *

class FullConnectedNeuralNetworkBprop(FullConnectedNeuralNetwork):
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

      weights_delta.append(dw)
      biases_delta.append(db)

      last_delta_valore = delta_valore

    # I delta dei pesi e i delta dei biases sono "al contrario", ovvero quelli in posizione 0 della lista in realtà sono per il layer di output, quindi faccio il
    # reverse della lista per averli sistemati
    weights_delta.reverse()
    biases_delta.reverse()

    return weights_delta, biases_delta