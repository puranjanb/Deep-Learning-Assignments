import numpy as np

class Constant:
    def __init__(self,constant_val=0.1):

        self.constant_val = constant_val

    def initialize(self,weights_shape,fan_in, fan_out):

        tensor = np.full(weights_shape,self.constant_val)

        return tensor

class UniformRandom:
    def __init__(self):
        None

    def initialize(self,weights_shape,fan_in, fan_out):

        weight_tensor = np.random.uniform(0, 1, (fan_in, fan_out))
        return weight_tensor

class Xavier:
    def __init__(self):
        None

    def initialize(self,weights_shape,fan_in, fan_out):

        sigma = np.sqrt(2/(fan_in+fan_out))

        weight_tensor = np.random.normal(0, sigma, weights_shape)
        return weight_tensor

class He:
    def __init__(self):
        None

    def initialize(self,weights_shape,fan_in, fan_out):

        sigma = np.sqrt(2/(fan_in))

        weight_tensor = np.random.normal(0, sigma, weights_shape)
        return weight_tensor
