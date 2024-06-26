import numpy as np
from abc import abstractmethod

class CostFunction():
  @abstractmethod
  def compute(output: np.ndarray, ground_truth: np.ndarray) -> float:
    raise NotImplementedError("Method not implemented")
  @abstractmethod
  def compute_derivate(output: np.ndarray, ground_truth: np.ndarray) -> float:
    raise NotImplementedError("Method not implemented")

class SumOfSquaredError(CostFunction):
  def compute(output: np.ndarray, ground_truth: np.ndarray) -> float:
    return 0.5 * np.sum(np.square((output - ground_truth)))
  def compute_derivate(output: np.ndarray, ground_truth: np.ndarray) -> float:
    return output - ground_truth

class MeanSquaredError(CostFunction):
  def compute(output: np.ndarray, ground_truth: np.ndarray) -> float:
    return (1 / output.shape[0]) * np.sum(np.square((output - ground_truth)))
  def compute_derivate(output: np.ndarray, ground_truth: np.ndarray) -> float:
    return (2 / output.shape[0]) * np.sum((output - ground_truth))

class CrossEntropy(CostFunction):
  def compute(output: np.ndarray, ground_truth: np.ndarray) -> float:
    observation_log = np.log(output, where=output>0)
    log_losses = -(ground_truth * observation_log)
    return np.sum(log_losses)
  def compute_derivate(output: np.ndarray, ground_truth: np.ndarray) -> float:
    return output - ground_truth