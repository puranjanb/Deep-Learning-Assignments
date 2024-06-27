class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)                                                                                  #float learning rate

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor

        self.weight_tensor = self.weight_tensor - self.learning_rate * self.gradient_tensor #calculating updated tensor weight
        return self.weight_tensor
