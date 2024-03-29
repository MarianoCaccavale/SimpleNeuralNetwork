from abc import ABC
from CostFunction import *
import numpy as np
from TrainingMethod.TrainingMethod import *
from CostFunction.CostFunction import *
from Net.Layer import Layer
from typing import List

import math


class FullConnectedNeuralNetwork(ABC):
  def __init__(self, layers: List[Layer], cost_function: CostFunction, learning_rate: int = 0.01):

    if len(layers) <= 0:
      raise Exception(f"Numero i layer sbagliato! Non può esistere una rete con {len(layers)} layers, il minimo è 3(1 di input, uno hidden e uno di output)")

    self.cost_function = cost_function

    # Controllo che i layer abbiano senso tra di loro, ovvero il layer i abbia un numero di pesi uguali al numero di neuroni del livello precedente(quindi sia full connected)
    for i in range(1, len(layers)):
      if layers[i].previous_layer_neurons != layers[i - 1].number_of_neurons:
        raise Exception(f"Layer {i} e {i-1} non compatibili! Il layer {i}esimo ho connessioni per {layers[i].previous_layer_neurons}, mentre il layer {i-1}esima ha {layers[i].number_of_neurons}")

    self.learning_rate = learning_rate

    self.layers = layers
    self.accuracy = 0
    # self.precision = 0
    # self.recall = 0

  def __feed_forward(self, input: np.ndarray) -> np.ndarray:
    if input.shape[0] != self.layers[0].previous_layer_neurons:
      raise Exception(f"Input di dimensione non corretta! La rete si aspetta in input un vettore di dimensione {self.layers[0].previous_layer_neurons} ma l'input fornito ha dimensioni {input.shape[0]}")

    previous_layer_outputs = input

    for layer in self.layers:
      previous_layer_outputs = layer.calculate_output(previous_layer_outputs)

    return previous_layer_outputs

  def __compute_validation_error(self, validation_set: np.ndarray, validation_targets: np.ndarray):
    error = 0
    accuracy = 0

    net_output = self.predict(validation_set)
    error = self.cost_function.compute(net_output, validation_targets)
    
    index_of_max_net_output = np.argmax(net_output, axis=0)
    index_of_max_ground_truth = np.argmax(validation_targets, axis=0)

    accuracy = np.where(index_of_max_net_output == index_of_max_ground_truth, 1, 0).sum()

    mean_error = error / (validation_set.shape[1])
    mean_accuracy = accuracy / (validation_set.shape[1])
    return mean_error, mean_accuracy
  
  def _back_propagate(self, ground_truth: np.ndarray):
    last_delta_valore = 0
    layer_index = 0

    weights_delta = []
    biases_delta = []

    for layer_index in range(len(self.layers)-1, -1, -1):
      layer = self.layers[layer_index]
      
      dZdW = self.layers[layer_index].X # 256, 1
      dAdZ = layer.activation_function_derivative # 10, 1

      # Output layer
      if layer_index == len(self.layers) - 1:
        dCdA = self.cost_function.compute_derivate(layer.A, ground_truth) # 10, 1
      else:
        dCdA = np.matmul(self.layers[layer_index + 1].weights.T, last_delta_valore)

      delta_valore = np.multiply(dAdZ, dCdA) #10, 1
      dw = np.matmul(delta_valore, dZdW.T)
      dw /= delta_valore.shape[1]

      db = dCdA.sum(axis = 1, keepdims=True)
      db /= delta_valore.shape[1]

      weights_delta.append(dw)
      biases_delta.append(db)

      last_delta_valore = delta_valore

    # I delta dei pesi e i delta dei biases sono "al contrario", ovvero quelli in posizione 0 della lista in realtà sono per il layer di output, quindi faccio il
    # reverse della lista per averli sistemati
    weights_delta.reverse()
    biases_delta.reverse()

    return weights_delta, biases_delta

  def train(self, training_set: np.ndarray, ground_truths: np.ndarray, epochs: int, training_method: TrainingMethod = TrainingMethod.BATCH, validation_set: np.ndarray = None, validation_targets: np.ndarray = None, batch_size: float = 0.1):

    if training_set.shape[1] != ground_truths.shape[1]:
      raise Exception(f"Training set e ground_truths non hanno lo stesso numero di campioni!")

    if validation_set is not None and validation_targets is not None:
      if validation_set.shape[1] != validation_targets.shape[1]:
        raise Exception(f"validation_set e validation_targets non hanno lo stesso numero di campioni!")

    if epochs <= 0:
      raise Exception("Il numero di epoch per l'addestramento deve essere almeno 1!")

    self.mean_train_error = []
    self.mean_val_error = []
    self.mean_val_accuracy = []
    self.accuracies = []

    number_of_samples = training_set.shape[1]
    number_of_val_samples = validation_set.shape[1] + 1 if validation_set is not None else 0
    print(f"Working with {number_of_samples} train samples and {number_of_val_samples} val samples")

    # L'index mi serve per sapere a che iterazione sono nel caso di batch o mini-batch
    index = 0
    # Il val error mi serve a sapere quanto sta performando la rete, epoca per epoca, su dati mai visti
    val_error = 0

    match training_method:
      # Caso medio-semplice: eseguo il feed forward su tutti i campioni che ho, facendo poi una media degli errori calcolati campione per campione. Questo errore medio viene
      # poi usato per aggiornare i pesi della rete
      case TrainingMethod.BATCH:
        for epoch in range(0, epochs):
          # Resetto l'errore per questa epoca
          error = 0
          index = 0

          # Mantengo due liste, in cui ad ogni indice trovo i delta di aggiornamento dell'i-esimo livello. Ad ogni training example sommo i delta, per poi farne la media
          # una volta finito il batch
          sum_of_weights_delta = []
          sum_of_biases_delta = []

          # Ad inizio di ogni epoca, uso la rete allo stato corrente per calcolarmi l'errore medio sul validation_set. In questo modo 
          # non vado in "conflitto" con l'aggiornamento dei pesi(calcolo il val_error PRIMA di aggiornare i pesi)
          if validation_set is not None:
              val_error, val_accuracy = self.__compute_validation_error(validation_set, validation_targets)
              self.mean_val_error.append(val_error)
              self.mean_val_accuracy.append(val_accuracy)

          # Calcolo l'output della rete
          net_output = self.__feed_forward(training_set)
          # Sommo gli errori campione per campione
          error = self.cost_function.compute(net_output, ground_truths)
          index_of_max_net_output = np.argmax(net_output, axis=0)
          index_of_max_ground_truth = np.argmax(ground_truths, axis=0)

          self.accuracy = np.where(index_of_max_net_output == index_of_max_ground_truth, 1, 0).sum()

          gradient_weight_list, gradient_bias_list = self._back_propagate(ground_truths)

          self._update_parameters(gradient_weight_list, gradient_bias_list)

          mean_error = error / number_of_samples
          self.mean_train_error.append(mean_error)
          self.accuracy = self.accuracy / number_of_samples
          self.accuracies.append(self.accuracy)

          if validation_set is not None:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - mean_val_error: {val_error:1.8f} - accuracy = {self.accuracy:1.8f} - val_accuracy = {val_accuracy:1.8f}")
          else:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - accuracy = {self.accuracy:1.3f}")

      # Caso difficile: devo dividere il training set in input in x mini-batch. Per ogni mini-batch poi devo calcolare la media dell'errore e usare la sua derivata per
      # aggiornare i pesi della rete
      case TrainingMethod.MINI_BATCH:

        current_batch_size = number_of_samples // (batch_size * number_of_samples)
        mini_training_batches = np.array_split(training_set, current_batch_size, axis = 1)
        mini_ground_truths_batches = np.array_split(ground_truths, current_batch_size, axis = 1)

        for epoch in range(0, epochs):
          error = 0
          val_error = 0

          for mini_training_batch, mini_ground_truth_batch in zip(mini_training_batches, mini_ground_truths_batches):

            number_of_samples_in_batch = mini_training_batch.shape[1]
            index = 0

            # Ad inizio di ogni epoca, uso la rete allo stato corrente per calcolarmi l'errore medio sul validation_set. In questo modo non vado in "conflitto" con
            # l'aggiornamento dei pesi(calcolo il val_error PRIMA di aggiornare i pesi)
            if validation_set is not None:
                val_error = self.__compute_validation_error(validation_set, validation_targets)
                self.mean_val_error.append(val_error)

            # Mantengo due liste, in cui ad ogni indice trovo i delta di aggiornamento dell'i-esimo livello. Ad ogni training example sommo i delta, per poi farne la media
            # una volta finito il batch
            sum_of_weights_delta = []
            sum_of_biases_delta = []

            for training_sample, ground_truth in zip(mini_training_batch.T, mini_ground_truth_batch.T):
              net_output = self.__feed_forward(training_sample)
              error = error + self.cost_function.compute(net_output, ground_truth)
              
              index_of_max_net_output = np.argmax(net_output)
              index_of_max_ground_truth = np.argmax(ground_truth)

              if index_of_max_net_output == index_of_max_ground_truth:
                self.accuracy = self.accuracy + 1

              if index == 0:
                sum_of_weights_delta, sum_of_biases_delta = self._back_propagate(ground_truth)
              else:
                weights_delta, biases_delta = self._back_propagate(ground_truth)
                for i in range(0, len(sum_of_weights_delta)):
                  sum_of_weights_delta[i] = sum_of_weights_delta[i] + weights_delta[i]
                  sum_of_biases_delta[i] = sum_of_biases_delta[i] + biases_delta[i]

              index = index + 1

            # Ora faccio la media dei delta
            for i in range(0, len(sum_of_weights_delta)):
              sum_of_weights_delta[i] = np.divide(sum_of_weights_delta[i], number_of_samples_in_batch)
              sum_of_biases_delta[i] = np.divide(sum_of_biases_delta[i], number_of_samples_in_batch)

            for layer_index in range(0, len(self.layers)):
              self.layers[layer_index].update_parameters(sum_of_weights_delta[layer_index], sum_of_biases_delta[layer_index])

          mean_error = error / number_of_samples
          self.mean_train_error.append(mean_error)
          self.accuracy = self.accuracy / number_of_samples
          self.accuracies.append(self.accuracy)
          if validation_set is not None:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - mean_val_error: {val_error:1.8f} - accuracy = {self.accuracy:1.8f}")
          else:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - accuracy = {self.accuracy:1.8f}")

      # Caso facile: per ogni campione calcolo la funzione di costo, la derivo e la uso per aggiornare i pesi della rete
      case TrainingMethod.ONLINE:
        for epoch in range(0, epochs):
          error = 0
          val_error = 0

          # Ad inizio di ogni epoca, uso la rete allo stato corrente per calcolarmi l'errore medio sul validation_set. In questo modo non vado in "conflitto" con
          # l'aggiornamento dei pesi(calcolo il val_error PRIMA di aggiornare i pesi)
          if validation_set is not None:
            val_error = self.__compute_validation_error(validation_set, validation_targets)
            self.mean_val_error.append(val_error)

          for training_sample, ground_truth in zip(training_set.T, ground_truths.T):
            net_output = self.__feed_forward(training_sample)
            error = error + self.cost_function.compute(net_output, ground_truth)

            index_of_max_net_output = np.argmax(net_output)
            index_of_max_ground_truth = np.argmax(ground_truth)

            if index_of_max_net_output == index_of_max_ground_truth:
              self.accuracy = self.accuracy + 1

            sum_of_weights_delta, sum_of_biases_delta = self._back_propagate(ground_truth)
            for layer_index in range(0, len(self.layers)):
              self.layers[layer_index].update_parameters(sum_of_weights_delta[layer_index], sum_of_biases_delta[layer_index])

          mean_error = error / number_of_samples
          self.mean_train_error.append(mean_error)
          self.accuracy = self.accuracy / number_of_samples
          self.accuracies.append(self.accuracy)
          if validation_set is not None:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - mean_val_error: {val_error:1.8f} - accuracy = {self.accuracy:1.8f}")
          else:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - accuracy = {self.accuracy:1.8f}")

    print("End training.")

  def _update_parameters(self, dw: List[np.ndarray], db: List[np.ndarray]):
    raise NotImplementedError()

  def predict(self, samples: np.ndarray) -> np.ndarray:
    return self.__feed_forward(samples)