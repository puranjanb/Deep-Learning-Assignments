from src_to_implement.Layers import Base
import numpy as np
from scipy import ndimage
import copy

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        self.sshape = stride_shape
        self.cshape = convolution_shape
        self.nkernels = num_kernels
        self.weights = np.random.uniform(0, 1, (self.nkernels, *self.cshape))

        if len(convolution_shape) == 2:
            self.padshape = (convolution_shape[1] - 1, 0)
        else:
            self.padshape = (convolution_shape[1] - 1, convolution_shape[2] - 1)

        self.bias = np.random.uniform(0, 1,self.nkernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self._optimizer_weights = None
        self._optimizer_bias = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_weights = copy.deepcopy(self.optimizer)
        self._optimizer_bias = copy.deepcopy(self.optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weights):
        self._gradient_weights = weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, bias):
        self._gradient_bias = bias

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.cshape[0] * self.cshape[1] * self.cshape[2]
        fan_out = self.nkernels * self.cshape[1] * self.cshape[2]
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        bsize = np.shape(input_tensor)[0]
        channel = input_tensor.shape[1]
        p1, p2 = int(self.padshape[0] / 2), int(self.padshape[1] / 2)
        if len(input_tensor.shape) == 3:
            new_tensor = np.zeros((bsize, channel, input_tensor.shape[2] + self.padshape[0]))
            if p1 == 0:
                new_tensor[:, :, :] = input_tensor[:, :, :]
            else:
                new_tensor[:, :, p1:-(self.padshape[0] - p1)] = input_tensor[:, :, :]

            p1 = int((input_tensor.shape[2] - self.cshape[1] + self.padshape[0]) / self.sshape[0]) + 1

            self.output_tensor = np.zeros((bsize, self.nkernels, p1))

            for b in range(bsize):
                for k in range(self.nkernels):
                    for row in range(p1):
                        window = new_tensor[b, :, row * self.sshape[0]:row * self.sshape[0] + self.cshape[1]]
                        self.output_tensor[b, k, row] = np.sum(np.multiply(self.weights[k, :, :], window)) + self.bias[k]
        else:
            new_tensor = np.zeros(
                (bsize, channel, input_tensor.shape[2] + self.padshape[0], input_tensor.shape[3] + self.padshape[1]))
            if p1 == 0 and p2 == 0:
                new_tensor[:, :, :, :] = input_tensor[:, :, :, :]
            else:
                new_tensor[:, :, p1:-(self.padshape[0] - p1), p2:-(self.padshape[1] - p2)] = input_tensor[:, :, :, :]

            p1 = int((input_tensor.shape[2] - self.cshape[1] + self.padshape[0]) / self.sshape[0]) + 1
            p2 = int((input_tensor.shape[3] - self.cshape[2] + self.padshape[1]) / self.sshape[1]) + 1

            self.output_tensor = np.zeros((bsize, self.nkernels, p1, p2))

            for b in range(bsize):
                for k in range(self.nkernels):
                    for row in range(p1):
                        for col in range(p2):
                            window = new_tensor[b, :, row * self.sshape[0]:row * self.sshape[0] + self.cshape[1],
                                     col * self.sshape[1]:col * self.sshape[1] + self.cshape[2]]
                            self.output_tensor[b, k, row, col] = np.sum(
                                np.multiply(self.weights[k, :, :, :], window)) + self.bias[k]
        return self.output_tensor

    def backward(self, error_tensor):
        bsize = np.shape(error_tensor)[0]
        channel = self.cshape[0]
        if len(error_tensor.shape) == 2:
            error_tensor = error_tensor.reshape(bsize,self.nkernels,*self.output_tensor.shape[2:])

        p1, p2 = int(self.padshape[0] / 2), int(self.padshape[1] / 2)
        nweights = np.zeros((self.cshape[0], self.nkernels, *self.cshape[1:]))
        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_bias = np.zeros(self.bias.shape)

        if len(error_tensor.shape) == 3:
            for k in range(self.nkernels):
                for c in range(self.cshape[0]):
                    nweights[c, k, :] = np.flip(self.weights[k, c, :])

            new_tensor = np.zeros((bsize, self.nkernels, self.input_tensor.shape[2] + self.padshape[0]))
            p1 = int((self.input_tensor.shape[2] + self.padshape[0] - error_tensor.shape[2] * self.sshape[0])/2)
            new_tensor[:, :, p1:p1+error_tensor.shape[2] * self.sshape[0]:self.sshape[0]] = error_tensor[:, :, :]

            rows = self.input_tensor.shape[2]
            output_tensor = np.zeros((bsize, channel, rows))

            for b in range(bsize):
                for c in range(channel):
                    for row in range(rows):
                        window = new_tensor[b, :, row: row + self.cshape[1]]
                        output_tensor[b, c, row] = np.sum(np.multiply(nweights[c, :, :], window))

            nerror_tensor = new_tensor[:,:,p1:p1+self.input_tensor.shape[2]]
            new_tensor = np.zeros((new_tensor.shape[0],self.cshape[0],new_tensor.shape[2]))
            p1 = int(self.padshape[0] / 2)
            if p1 == 0:
                new_tensor[:, :, :] = self.input_tensor[:, :, :]
            else:
                new_tensor[:, :, p1:-(self.padshape[0] - p1)] = self.input_tensor[:, :, :]

            for k in range(self.nkernels):
                for c in range(channel):
                    for row in range(self.cshape[1]):
                        window = new_tensor[:, c, row: row + rows]
                        self.gradient_weights[k,c,row] += np.sum(np.multiply(window,nerror_tensor[:,k,:]))
                    self.gradient_bias[k] += np.sum(np.multiply(self.input_tensor[:,c,:],nerror_tensor[:,k,:]))
                    #self.gradient_weights[k, c, :] = np.flip(self.gradient_weights[k,c,:])

        else:
            for k in range(self.nkernels):
                for c in range(self.cshape[0]):
                    nweights[c, k, :, :] = np.rot90(self.weights[k, c, :, :],2)

            new_tensor = np.zeros((bsize, self.nkernels, self.input_tensor.shape[2] + self.padshape[0],
                                   self.input_tensor.shape[3] + self.padshape[1]))

            p1 = int((self.input_tensor.shape[2] + self.padshape[0] - error_tensor.shape[2] * self.sshape[0]) / 2)
            p2 = int((self.input_tensor.shape[3] + self.padshape[1] - error_tensor.shape[3] * self.sshape[1]) / 2)
            new_tensor[:, :, p1:p1+error_tensor.shape[2] * self.sshape[0]:self.sshape[0],
            p2:p2+error_tensor.shape[3] * self.sshape[1]:self.sshape[1]] = error_tensor[:, :, :, :]

            rows = self.input_tensor.shape[2]
            cols = self.input_tensor.shape[3]
            output_tensor = np.zeros((bsize, channel, rows, cols))
            for b in range(bsize):
                for c in range(channel):
                    for row in range(rows):
                        for col in range(cols):
                            window = new_tensor[b, :, row: row + self.cshape[1], col: col + self.cshape[2]]
                            output_tensor[b, c, row, col] = np.sum(np.multiply(nweights[c, :, :, :], window))

            nerror_tensor = new_tensor[:, :, p1:p1 + self.input_tensor.shape[2], p2:p2 + self.input_tensor.shape[3]]
            new_tensor = np.zeros((new_tensor.shape[0],self.cshape[0],*new_tensor.shape[2:]))
            p1, p2 = int(self.padshape[0] / 2), int(self.padshape[1] / 2)
            if p1 == 0 and p2 == 0:
                new_tensor[:, :, :, :] = self.input_tensor[:, :, :, :]
            else:
                new_tensor[:, :, p1:-(self.padshape[0] - p1), p2:-(self.padshape[1] - p2)] = self.input_tensor[:, :, :,:]

            for k in range(self.nkernels):
                for c in range(channel):
                    for row in range(self.cshape[1]):
                        for col in range(self.cshape[2]):
                            window = new_tensor[:, c, row: row + rows, col: col + cols]
                            self.gradient_weights[k, c, row, col] += np.sum(np.multiply(window, nerror_tensor[:, k, :, :]))
        if len((self.input_tensor).shape) == 4:
            self.gradient_bias = np.sum(error_tensor, axis = (0,2,3))
        else:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))
        if self._optimizer_weights is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)

        if self._optimizer_bias is not None:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)
        return output_tensor
