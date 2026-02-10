import numpy as np

class Layer_Dense:
   def __init__(self, n_inputs, n_neurons):
      self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
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

class Activation_Linear:
   def __init__(self):
      self.alpha = 1

   def forward(self, inputs):
      self.inputs = inputs
      self.output = self.alpha * inputs

   def backward(self, dvalues):
      self.dinputs = dvalues.copy()

class Activation_Tanh:
   def forward(self, inputs):
      self.inputs = inputs
      self.output = np.tanh(inputs)

   def backward(self, dvalues):
      self.dinputs = dvalues * (1 - self.output ** 2)

class Cemantix_Bot:
   def __init__(self, history_size=10):
      self.history_size = history_size

      input_dim = history_size * 501 # words are in 500 dimension and 1 dimension for the score
      self.layer1 = Layer_Dense(input_dim, 2048)
      self.activation1 = Activation_Linear()

      self.layer2 = Layer_Dense(2048, 1024)
      self.activation2 = Activation_Tanh()

      self.layer3 = Layer_Dense(1024, 500)

   def forward(self, inputs):
      self.layer1.forward(inputs)
      self.activation1.forward(self.layer1.output)
      self.layer2.forward(self.activation1.output)
      self.activation2.forward(self.layer2.output)
      self.layer3.forward(self.activation2.output)
      return self.layer3.output

   def backward(self, dvalues, learning_rate):
      self.layer3.backward(dvalues, learning_rate)
      self.activation2.backward(self.layer3.dinputs)
      self.layer2.backward(self.activation2.dinputs, learning_rate)
      self.activation1.backward(self.layer2.dinputs)
      self.layer1.backward(self.activation1.dinputs, learning_rate)

   def save_model(self, filename="neural_network.npz"):
      np.savez(filename,
               l1_w=self.layer1.weights,
               l1_b=self.layer1.biases,
               l2_w=self.layer2.weights,
               l2_b=self.layer2.biases,
               l3_w=self.layer3.weights,
               l3_b=self.layer3.biases)

   def load_model(self, filename="neural_network.npz"):
      try:
         data = np.load(filename)
         self.layer1.weights = data["l1_w"]
         self.layer1.biases = data["l1_b"]
         self.layer2.weights = data["l2_w"]
         self.layer2.biases = data["l2_b"]
         self.layer3.weights = data["l3_w"]
         self.layer3.biases = data["l3_b"]
      except FileNotFoundError:
          print("No saved brain found")
