import numpy as np
from src_to_implement.Layers import Base

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = np.copy(input_tensor)
        input_tensor = np.maximum(0, input_tensor)              #replacing value of input tensor with maximum of 0 or input tensor value
        return input_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.error_tensor[self.input_tensor <= 0] = 0                 #storing error tensor values as 0 where input tensor was negetive
        return self.error_tensor