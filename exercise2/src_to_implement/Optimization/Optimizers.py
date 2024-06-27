import  numpy as np

class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor

        self.weight_tensor = self.weight_tensor - self.learning_rate * self.gradient_tensor
        return self.weight_tensor

class SgdWithMomentum:
    def __init__(self, learning_rate,momentum):
        self.learning_rate = float(learning_rate)
        self.momentum = momentum
        self.v = 0


    def calculate_update(self, weight_tensor, gradient_tensor):

        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor

        self.v = self.momentum * self.v - self.learning_rate*gradient_tensor


        self.weight_tensor = self.weight_tensor + self.v


        return self.weight_tensor

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = float(learning_rate)
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.t += 1
        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor
        self.v = self.mu*self.v + (1-self.mu)*gradient_tensor
        self.r = self.r*self.rho + (1-self.rho)*np.square(gradient_tensor)
        vh = self.v/(1-self.mu**self.t)
        rh = self.r/(1-self.rho**self.t)
        self.weight_tensor = self.weight_tensor - self.learning_rate * vh / (np.sqrt(rh) + np.finfo(float).eps)
        return self.weight_tensor





