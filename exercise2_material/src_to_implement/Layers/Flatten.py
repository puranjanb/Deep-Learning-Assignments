import numpy as np
class Flatten:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return input_tensor.reshape((np.shape(input_tensor)[0]),np.prod(np.shape(input_tensor)[1:]))

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_tensor.shape)