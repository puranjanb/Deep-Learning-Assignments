import copy as cp


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = []
        self.loss_layer = []

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()

        for i in range(len(self.layers)):
            self.input_tensor = self.layers[i].forward(self.input_tensor)
        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)        #computing total loss
        return loss

    def backward(self):
        self.error_tensor = self.loss_layer.backward(self.label_tensor)
        for i in reversed(range(len(self.layers))):
            self.error_tensor = self.layers[i].backward(self.error_tensor)              #traversing layers backward


    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = cp.deepcopy(self.optimizer)
        self.layers.append(layer)                                                                                       #appending layers

    def train(self, iterations):
        for i in range(iterations):
            self.loss.append(self.forward())                                                                        #storing loss for each layer
            self.backward()

    def test(self, input_tensor):
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)                                     #computes total loss after going through all layers
        return input_tensor

