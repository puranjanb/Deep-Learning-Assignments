import numpy as np
import matplotlib.pyplot as plt


class Checker:
    output = []

    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        if (self.resolution % (2 * self.tile_size) == 1):
            print("incorrect resolution")
            return
        zo = np.zeros(self.tile_size * self.tile_size).reshape(self.tile_size, self.tile_size)
        on = np.ones(self.tile_size * self.tile_size).reshape(self.tile_size, self.tile_size)
        x = np.concatenate((zo, on), axis=1)
        y = np.concatenate((on, zo), axis=1)
        z = np.concatenate((x, y), axis=0)
        size = int(self.resolution / (2 * self.tile_size))
        self.output = np.tile(z, (size, size))
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle:
    output = []

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        x = np.linspace(0, self.resolution - 1, self.resolution)
        y = np.linspace(0, self.resolution - 1, self.resolution)
        x, y = np.meshgrid(x, y)
        x_0 = self.position[0]
        y_0 = self.position[1]
        mask = np.sqrt(np.square(x - x_0) + np.square(y - y_0))
        a = np.ma.masked_less_equal(mask, self.radius)
        b = np.ma.getmask(a)
        self.output = b
        return np.copy(b)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:
    output = []

    def __init__(self, resolution):
        self.resolution = resolution

    def draw(self):
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        r, g = np.meshgrid(x, y)
        b = np.flip(r)
        rgb = np.zeros((self.resolution, self.resolution, 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        self.output = rgb
        return np.copy(rgb)

    def show(self):
        plt.imshow(self.output)
        plt.show()