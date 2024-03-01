import numpy as np
from abc import abstractmethod

class CostFunction():
  @abstractmethod
  def compute(output: np.array, ground_truth: np.array) -> float:
    raise NotImplementedError("Method not implemented")
  @abstractmethod
  def compute_derivate(output: np.array, ground_truth: np.array) -> float:
    raise NotImplementedError("Method not implemented")

class SumOfSquaredError(CostFunction):
  def compute(output: np.array, ground_truth: np.array) -> float:
    return 0.5 * np.sum(np.square((output - ground_truth)))
  def compute_derivate(output: np.array, ground_truth: np.array) -> float:
    return output - ground_truth

class MeanSquaredError(CostFunction):
  def compute(output: np.array, ground_truth: np.array) -> float:
    return (1 / output.shape[0]) * np.sum(np.square((output - ground_truth)))
  def compute_derivate(output: np.array, ground_truth: np.array) -> float:
    return (2 / output.shape[0]) * np.sum((output - ground_truth))

class CrossEntropy(CostFunction):
  def compute(output: np.array, ground_truth: np.array) -> float:
    observation_log = np.log(output)
    log_losses = -np.multiply(ground_truth, observation_log)
    return np.sum(log_losses)
  def compute_derivate(output: np.array, ground_truth: np.array) -> float:
    return output - ground_truth