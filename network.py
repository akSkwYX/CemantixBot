import numpy as np

class Layer_Dense:
   def __init__(self, n_inputs, n_neurons):
      self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
      self.biases = np.zeros((1, n_neurons))

   def forward(self, inputs):
      self.inputs = inputs
      self.output = np.dot(inputs, self.weights) + self.biases

   def backward(self, dvalues, learning_rate):
      self.dweights = np.dot(self.inputs.T, dvalues)
      self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
      self.dinputs = np.dot(dvalues, self.weights.T)

      self.weights -= learning_rate * self.dweights
      self.biases -= learning_rate * self.dbiases


class Activation_ReLU:
   def forward(self, inputs):
      self.inputs = inputs
      self.output = np.maximum(0, inputs)

   def backward(self, dvalues):
      self.dinputs = dvalues.copy()
      self.dinputs[self.inputs <= 0] = 0

class Cemantix_Bot:
    def __init__(self):
        self.layer1 = Layer_Dense(500, 128)
        self.activation1 = Activation_ReLU()

        self.layer2 = Layer_Dense(128, 500)

    def forward(self, inputs):
        self.layer1.forward(inputs)
        self.activation1.forward(self.layer1.output)
        self.layer2.forward(self.activation1.output)
        return self.layer2.output

    def backward(self, dvalues, learning_rate):
        self.layer2.backward(dvalues, learning_rate)
        self.activation1.backward(self.layer2.dinputs)
        self.layer1.backward(self.activation1.dinputs, learning_rate)
