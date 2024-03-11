import numpy as np
from abc import abstractmethod

class ActivationFunction():
  @abstractmethod
  def compute(x:np.array) -> np.array:
    raise NotImplementedError("Method not implemented")
  @abstractmethod
  def compute_derivate(x:np.array) -> np.array:
    raise NotImplementedError("Method not implemented")

class IdentityActivationFunction(ActivationFunction):
  def compute(x:np.array) -> np.array:
    return x
  def compute_derivate(x:np.array) -> np.array:
    return np.full_like(x, 1)

class SoftMaxActivationFunction(ActivationFunction):
  # Given the fact that np.exp tent to overflow, we use the stable softmax
  def compute(x:np.array) -> np.array:
    x_exp:np.float256=np.exp(x-np.max(x))
    return x_exp/np.sum(x_exp)
  def compute_derivate(x:np.array) -> np.array:
    return np.full_like(x, 1)
  """def compute_derivate(x:np.array) -> np.array:
    gradient = np.empty((x.shape[0], x.shape[0]))
    softmax_values = SoftMaxActivationFunction.compute(x)
    for i in range(0, softmax_values.shape[0]):
      for j in range(0, x.shape[0]):
        if i == j:
          gradient[i] = softmax_values[i] * (1-x[j])
        else:
          gradient[i] = -softmax_values[i]*x[j]
    return np.array(gradient)"""

class SigmoidActivationFunction(ActivationFunction):
  def compute(x:np.array) -> np.array:
    return 1 / (1 + np.exp(-x))
  def compute_derivate(x:np.array) -> np.array:
    sigmoid_of_x = 1 / (1 + np.exp(-x))
    return sigmoid_of_x * (1 - sigmoid_of_x)

class TanhActivationFunction(ActivationFunction):
  def compute(x:np.array) -> np.array:
    return np.tanh(x)
  def compute_derivate(x:np.array) -> np.array:
    tanh_of_x = np.tanh(x)
    return 1  - (tanh_of_x**2)

class RELUActivationFunction(ActivationFunction):
  def compute(x:np.array) -> np.array:
    result = []
    for elem in x:
      result.append(max(0.0, elem))
    return np.array(result)
  def compute_derivate(x:np.array) -> np.array:
    result = []
    for elem in x:
      result.append(0 if elem <= 0 else 1)
    return np.array(result)

class LeakyRELUActivationFunction(ActivationFunction):
  def compute(x:np.array) -> np.array:
    result = []
    for elem in x:
      result.append(max(0.01*elem, elem))
    return np.array(result)
  def compute_derivate(x:np.array) -> np.array:
    result = []
    for elem in x:
      result.append(0.01 if elem <= 0 else 1)
    return np.array(result)