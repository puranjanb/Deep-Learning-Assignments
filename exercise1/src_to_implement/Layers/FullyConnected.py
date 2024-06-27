import numpy as np

from src_to_implement.Layers import Base

class FullyConnected(Base.BaseLayer):       #inheriting parent class BaseLayer
    def __init__(self, input_size, output_size):
        super().__init__()                                             #calling parent class constructor
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (self.input_size + 1, self.output_size))             #initialising weight
        self.optimizer = None

    def forward(self, input_tensor):
       self.input_tensor = input_tensor
       self.input_tensor = np.ones((input_tensor.shape[0], input_tensor.shape[1] + 1))     #adding one col for bias
       self.input_tensor[:, :-1] = input_tensor                                                                                      #initialising input except last col
       self.output_tensor = np.dot(self.input_tensor, self.weights)                                             #computing dot product
       return self.output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.last_error = np.dot(self.error_tensor, self.weights[:-1,:].T)                  #computing last error
        self.gradient_weights = np.dot(self.input_tensor.T, self.error_tensor)     #computing gradient for new weights

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)         #computing new weights if optimizer has any value

        return self.last_error