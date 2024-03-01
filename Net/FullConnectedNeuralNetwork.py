from abc import ABC
from CostFunction import *
import Layer
import numpy as np
from TrainingMethod.TrainingMethod import *
from CostFunction.CostFunction import *
from Layer import *
from typing import List


class FullConnectedNeuralNetwork(ABC):
  def __init__(self, layers: List[Layer], cost_function: CostFunction, learning_rate: int = 0.01):

    if len(layers) <= 0:
      raise Exception(f"Numero i layer sbagliato! Non può esistere una rete con {len(layers)} layers, il minimo è 3(1 di input, uno hidden e uno di output)")

    self.cost_function = cost_function

    # Controllo che i layer abbiano senso tra di loro, ovvero il layer i abbia un numero di pesi uguali al numero di neuroni del livello precedente(quindi sia full connected)
    for i in range(1, len(layers)):
      if layers[i].previous_layer_neurons != layers[i - 1].number_of_neurons:
        raise Exception(f"Layer {i} e {i-1} non compatibili! Il layer {i}esimo ho connessioni per {layers[i].previous_layer_neurons}, mentre il layer {i-1}esima ha {layers[i].number_of_neurons}")

    for layer in layers:
      layer.learning_rate = learning_rate

    self.layers = layers
    self.accuracy = 0
    # self.precision = 0
    # self.recall = 0

  def __feed_forward(self, input: np.array) -> np.array:
    if input.shape[0] != self.layers[0].previous_layer_neurons:
      raise Exception(f"Input di dimensione non corretta! La rete si aspetta in input un vettore di dimensione {self.layers[0].previous_layer_neurons} ma l'input fornito ha dimensioni {input.shape[0]}")

    previous_layer_outputs = input

    for layer in self.layers:
      previous_layer_outputs = layer.calculate_output(previous_layer_outputs)

    return previous_layer_outputs

  def __compute_validation_error(self):
    error = 0
    for sample, ground_truth in zip(self.validation_set.T, self.validation_targets.T):
      net_output = self.predict(sample)
      index_of_max_net_output = np.argmax(net_output)
      index_of_max_ground_truth = np.argmax(ground_truth)
      error = error + self.cost_function.compute(net_output, ground_truth)

    mean_error = error / self.validation_set.shape[1]
    return mean_error

  def train(self, training_set: np.array, ground_truths: np.array, validation_set: np.array, validation_targets: np.array, epochs: int, training_method: TrainingMethod = TrainingMethod.BATCH, batch_size: float = 0.1):

    if training_set.shape[1] != ground_truths.shape[1]:
      raise Exception(f"Training set e ground_truths non hanno lo stesso numero di campioni!")

    if validation_set is not None and validation_targets is not None:
      if validation_set.shape[1] != validation_targets.shape[1]:
        raise Exception(f"validation_set e validation_targets non hanno lo stesso numero di campioni!")

      self.validation_set = validation_set
      self.validation_targets = validation_targets

    if epochs <= 0:
      raise Exception("Il numero di epoch per l'addestramento deve essere almeno 1!")

    number_of_samples = training_set.shape[1]
    print(f"Working with {number_of_samples} samples")

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

          # Ad inizio di ogni epoca, uso la rete allo stato corrente per calcolarmi l'errore medio sul validation_set. In questo modo non vado in "conflitto" con l'aggiornamento
          # dei pesi(calcolo il val_error PRIMA di aggiornare i pesi)
          if self.validation_set is not None:
              val_error = self.__compute_validation_error()

          # Per tutto il training set
          for training_sample, ground_truth in zip(training_set.T, ground_truths.T):
            # Calcolo l'output della rete
            net_output = self.__feed_forward(training_sample)
            # Sommo gli errori campione per campione
            error = error + self.cost_function.compute(net_output, ground_truth)
            error_gradient = self.cost_function.compute_derivate(net_output, ground_truth)

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
            mean_function = lambda x: x/number_of_samples
            vectorize_mean_function = np.vectorize(mean_function)
            sum_of_weights_delta[i] = vectorize_mean_function(sum_of_weights_delta[i])
            sum_of_biases_delta[i] = vectorize_mean_function(sum_of_biases_delta[i])

          for layer_index in range(0, len(self.layers)):
            self.layers[layer_index].update_parameters(sum_of_weights_delta[layer_index], sum_of_biases_delta[layer_index])

          mean_error = error / number_of_samples
          self.accuracy = self.accuracy / number_of_samples
          if self.validation_set is not None:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - mean_val_error: {val_error:1.8f}- accuracy = {self.accuracy:1.8f}")
          else:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - accuracy = {self.accuracy:1.8f}")

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
            if self.validation_set is not None:
                val_error = self.__compute_validation_error()

            # Mantengo due liste, in cui ad ogni indice trovo i delta di aggiornamento dell'i-esimo livello. Ad ogni training example sommo i delta, per poi farne la media
            # una volta finito il batch
            sum_of_weights_delta = []
            sum_of_biases_delta = []

            for training_sample, ground_truth in zip(mini_training_batch.T, mini_ground_truth_batch.T):
              net_output = self.__feed_forward(training_sample)
              error = error + self.cost_function.compute(net_output, ground_truth)
              error_gradient = self.cost_function.compute_derivate(net_output, ground_truth)

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
          self.accuracy = self.accuracy / number_of_samples
          if self.validation_set is not None:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - mean_val_error: {val_error:1.8f}- accuracy = {self.accuracy:1.8f}")
          else:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - accuracy = {self.accuracy:1.8f}")

      # Caso facile: per ogni campione calcolo la funzione di costo, la derivo e la uso per aggiornare i pesi della rete
      case TrainingMethod.ONLINE:
        for epoch in range(0, epochs):
          error = 0
          val_error = 0

          # Ad inizio di ogni epoca, uso la rete allo stato corrente per calcolarmi l'errore medio sul validation_set. In questo modo non vado in "conflitto" con
          # l'aggiornamento dei pesi(calcolo il val_error PRIMA di aggiornare i pesi)
          if self.validation_set is not None:
            mean_val_error = self.__compute_validation_error()

          for training_sample, ground_truth in zip(training_set.T, ground_truths.T):
            net_output = self.__feed_forward(training_sample)
            error = error + self.cost_function.compute(net_output, ground_truth)
            error_gradient = self.cost_function.compute_derivate(net_output, ground_truth)

            if self.validation_set is not None:
              val_error = val_error + self.__compute_validation_error()

            index_of_max_net_output = np.argmax(net_output)
            index_of_max_ground_truth = np.argmax(ground_truth)

            if index_of_max_net_output == index_of_max_ground_truth:
              self.accuracy = self.accuracy + 1

            sum_of_weights_delta, sum_of_biases_delta = self._back_propagate(ground_truth)
            for layer_index in range(0, len(self.layers)):
              self.layers[layer_index].update_parameters(sum_of_weights_delta[layer_index], sum_of_biases_delta[layer_index])

          mean_error = error / number_of_samples
          self.accuracy = self.accuracy / number_of_samples
          if self.validation_set is not None:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - mean_val_error: {val_error:1.8f}- accuracy = {self.accuracy:1.8f}")
          else:
            print(f"Fine epoch #{epoch}; mean_train_error = {mean_error:1.8f} - accuracy = {self.accuracy:1.8f}")

    print("End training.")

  def _back_propagate(self, ground_truth: np.array):
    raise NotImplementedError("STAI USANDO UNA CLASSE ASTRATTA SOCIO, ACCANNA")

  def predict(self, sample: np.array) -> np.array:
    return self.__feed_forward(sample)