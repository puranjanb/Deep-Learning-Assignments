import numpy as np
class Pooling:
    def __init__(self,stride_shape,pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.location_tensor = np.zeros(self.input_tensor.shape)
        batch_size = self.input_tensor.shape[0]
        n_channels = self.input_tensor.shape[1]
        height = self.input_tensor.shape[2]
        width = self.input_tensor.shape[3]
        height_output = int((height - self.pooling_shape[0])/self.stride_shape[0])+1
        width_output = int((width - self.pooling_shape[1])/self.stride_shape[1])+1
        self.output_tensor = np.zeros((batch_size,n_channels,height_output,width_output))
        self.indices = []

        for bs in range(batch_size):
            for ch in range(n_channels):
                for ho in range(height_output):
                    for wo in range(width_output):
                        x_start = wo * self.stride_shape[1]
                        x_stop = x_start + self.pooling_shape[1]
                        y_start = ho * self.stride_shape[0]
                        y_stop = y_start + self.pooling_shape[0]
                        pooling = self.input_tensor[bs,ch,y_start:y_stop,x_start:x_stop]
                        z = pooling == np.max(pooling.flatten(), keepdims=False)
                        index = np.unravel_index(np.argmax(pooling.flatten()),pooling.shape)
                        index = (bs, ch, index[0] + y_start, index[1] + x_start)
                        self.indices.append(index)
                        self.output_tensor[bs,ch,ho,wo] = np.max(pooling)
                        self.location_tensor[bs,ch,y_start:y_stop,x_start:x_stop] += z
        return self.output_tensor

    def backward(self,error_tensor):
        self.error_tensor = error_tensor
        nerror_tensor = np.zeros((self.input_tensor.shape))
        i,x,y = 0,0,0
        for ind in self.indices:
            i,x,y=0,0,0
            while(i == 0):
                if self.input_tensor[ind] == self.output_tensor[ind[0],ind[1],x,y] and self.location_tensor[ind] ==1:
                    nerror_tensor[ind] += error_tensor[ind[0],ind[1],x,y]
                    i = 1
                elif self.input_tensor[ind] == self.output_tensor[ind[0],ind[1],x,y] and self.location_tensor[ind] ==2:
                    nerror_tensor[ind] += error_tensor[ind[0],ind[1],x,y]
                    self.output_tensor[ind[0], ind[1], x, y] = -self.input_tensor[ind]
                    i = 1
                x += 1
                if x == self.output_tensor.shape[2]:
                    y += 1
                    x = 0
                if y == self.output_tensor.shape[3]:
                    i=1
        return nerror_tensor
