import numpy as np
from src_to_implement.Layers import Base

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        numerator = np.exp(self.input_tensor - np.max(self.input_tensor))               #shifting input tensor values to reduce magnitude of values
        self.probability = numerator/np.array([numerator.sum(axis=1)]).T               #computing probability by dividing each value by sum of all values along column
        return self.probability

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.error_tensor = self.probability*(self.error_tensor - np.sum(np.multiply(self.error_tensor, self.probability),axis=1, keepdims=True))
        #calculating error using probability and error tensor
        return self.error_tensor
