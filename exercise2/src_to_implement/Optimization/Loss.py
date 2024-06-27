import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass
    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        loss = np.sum(-np.log(self.input_tensor[np.where(self.label_tensor == 1)] + np.finfo(float).eps))
        #computing loss as sum of log of values for each row and randor column of input tensor
        return loss

    def backward(self, label_tensor):
        self.label_tensor = label_tensor
        self.error_tensor = - self.label_tensor/self.input_tensor           #computing error tensor
        return self.error_tensor
