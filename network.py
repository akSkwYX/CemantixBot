from typing import final
from typing import Callable
from typing import Any
import numpy as np

# Probability distributions for weight initialization

@final
class Distribution:
    def __init__(self, fun : Callable[[int, int], np.ndarray]):
        self.fun = fun

    def init(self, input_length : int, output_length : int) -> np.ndarray:
        return self.fun(input_length, output_length)

Uniform = Distribution(lambda input_length, output_length: np.random.uniform(-1, 1, (output_length, input_length)))
Normal = Distribution(lambda input_length, output_length: np.random.normal(0, 1, (output_length, input_length)))
Xavier = Distribution(lambda input_length, output_length: np.random.normal(0, np.sqrt(2 / (input_length + output_length)), (output_length, input_length)))
He = Distribution(lambda input_length, output_length: np.random.normal(0, np.sqrt(2 / input_length), (output_length, input_length)))

# Activation functions

@final
class Activation:
    def __init__(self, fun : Callable[[np.ndarray], np.ndarray], derivative : Callable[[np.ndarray], np.ndarray]):
        self.f = fun
        self.df = derivative

ReLU = Activation(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))
TanH = Activation(lambda x: np.tanh(x), lambda x: 1 - np.tanh(x) ** 2)
Sigmoid = Activation(lambda x: 1 / (1 + np.exp(-x)), lambda x: np.exp(-x) / ((1 + np.exp(-x)) ** 2))
Linear = Activation(lambda x: x, lambda x: np.ones_like(x))
SoftMax = Activation(lambda x: np.exp(x) / np.sum(np.exp(x)), lambda x: np.exp(x) * (np.sum(np.exp(x)) - np.exp(x)) / (np.sum(np.exp(x)) ** 2))

# Cost functions

@final
class Cost:
    def __init__(self, fun : Callable[[np.ndarray, np.ndarray], float], derivative : Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.f = fun
        self.df = derivative

MSE = Cost(lambda y_true, y_pred: float(np.mean((y_true - y_pred) ** 2)),
           lambda y_true, y_pred: 2 * (y_pred - y_true) / y_true.size)
CrossEntropy = Cost(lambda y_true, y_pred: -np.sum(y_true * np.log(y_pred + 1e-15)) / y_true.shape[0],
                    lambda y_true, y_pred: -y_true / (y_pred + 1e-15) / y_true.shape[0])
SEL = Cost(lambda y_true, y_pred: np.sum((y_true - y_pred) ** 2),
           lambda y_true, y_pred: 2 * (y_pred - y_true))

# Dense Layers

@final
class Dense_Layer:
    def __init__(self, input_length : int, output_length : int, distribution : Distribution, activation : Activation):
        self.input_length = input_length
        self.input : np.ndarray = np.zeros(input_length)
        self.output_length = output_length
        self.output : np.ndarray = np.zeros(output_length)
        self.activation = activation
        self.weights : np.ndarray = distribution.init(input_length, output_length)
        self.biases : np.ndarray = np.zeros(output_length)

    def forward(self, inputs : np.ndarray):
        self.input = inputs
        self.output = self.activation.f(np.matmul(self.weights, inputs) + self.biases)

    def backward(self, dvalues : np.ndarray, learning_rate : float):
        self.weights -= learning_rate * np.matmul(dvalues * self.activation.df(self.output), self.input.T)
        self.biases -= learning_rate * dvalues * self.activation.df(self.output)
        
# Neural Network

@final
class Neural_Network:
    def __init__(self, layers : list[Dense_Layer], cost : Cost, output_fun : Callable[[np.ndarray], Any]):
        self.layers = layers
        self.cost = cost
        self.output_fun = output_fun

    def forward(self, inputs : np.ndarray):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
        return self.output_fun(inputs)

    def backward(self, target : np.ndarray, learning_rate : float):
        dvalues = self.cost.df(self.layers[-1].output, target)
        for layer in reversed(self.layers):
            layer.backward(dvalues, learning_rate)
            dvalues = np.matmul(layer.weights.T, dvalues * layer.activation.df(layer.output))

    def predict(self, X : np.ndarray) -> np.ndarray:
        return self.forward(X)

    def train(self, X : np.ndarray, Y : np.ndarray, success_rate : float, max_epochs : int, learning_rate : float):
        """
        X : inputs data,
        Y : expected output data for each input,
        success_rate : percentage of correct predictions required to stop training,
        max_epochs : maximum number of training iterations,
        learning_rate : step size for weight updates
        """
        for epoch in range(max_epochs):
            errors = 0
            for x, y in zip(X, Y):
                if self.predict(x) != y:
                    errors += 1
                self.backward(y, learning_rate)
            if errors / len(X) <= 1 - success_rate:
                print(f"Training stopped at epoch {epoch + 1}")
                break
            print(f"Epoch {epoch + 1}: {100 * (1 - errors / len(X)):.2f}% success rate")

    def tests(self, X : np.ndarray, Y : np.ndarray) -> float:
        correct = 0
        for x, y in zip(X, Y):
            if self.predict(x) == y:
                correct += 1
        return correct / len(X)

    def save(self, filename : str):
        np.savez(filename, layers=[(layer.weights, layer.biases) for layer in self.layers])

    def load(self, filename : str):
        data = np.load(filename, allow_pickle=True)
        for layer, (weights, biases) in zip(self.layers, data['layers']):
            layer.weights = weights
            layer.biases = biases
