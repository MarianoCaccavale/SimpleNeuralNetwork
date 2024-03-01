import numpy as np
import ActivationFunction

class Layer():
  def __init__(self, previous_layer_neurons: int, number_of_neurons: int, activation_function: ActivationFunction, random_initialize_weights: bool = True, random_initialize_bias: bool = True, learning_rate: float = 0.01):

    if previous_layer_neurons == 0 or number_of_neurons == 0:
      raise Exception("Valori errati per i parametri del layer! Non può esistere un layer con 0 neuroni o che non riceve connessioni da nessuno")

    self.number_of_neurons = number_of_neurons
    self.previous_layer_neurons = previous_layer_neurons

    # Inizializzo i pesi con un range molto piccolo per tamponare il problema di "z" calcolati da ogni neurone molto grandi o molto piccoli, cosa che rende
    # rende il training molto lento. Si potrebbe pensare di introdurre la regolarizzazione dei pesi, ma è un improvement lasciato per altri momenti
    weights_matrix_shape = (number_of_neurons, previous_layer_neurons)
    self.weights = np.random.uniform(low = -0.1, high = 0.1, size = weights_matrix_shape) if random_initialize_weights else np.ones(weights_matrix_shape)

    bias_vector_shape = (number_of_neurons,)
    self.biases = np.random.uniform(low = -1, high= 1, size = bias_vector_shape) if random_initialize_bias else np.zeros(bias_vector_shape)
    self.learning_rate = learning_rate

    self.activation_function = activation_function

  def update_parameters(self, dw: np.array, db: np.array):
    self.weights = self.weights - (dw * self.learning_rate)
    self.biases = self.biases[:, np.newaxis] - (db * self.learning_rate)
    self.biases = self.biases.flatten()

  def _calculate_Z(self, previous_layer_outputs: np.array, bias: np.array) -> np.array:
    self.X = previous_layer_outputs
    self.Z = np.dot(self.weights, previous_layer_outputs)
    self.Z = self.Z + bias
    return self.Z

  def calculate_output(self, previous_layer_outputs: np.array) -> np.array:

    Z = self._calculate_Z(previous_layer_outputs, self.biases)
    self.A = self.activation_function.compute(Z)
    self.activation_function_derivative = self.activation_function.compute_derivate(self.Z)
    return self.A
