import numpy as np
from scipy import ndimage
a = np.array(range(75))
a = a.reshape(3,5,5)
b = np.zeros((3,3,3))
b[:,::2,::2] = 1
print(ndimage.convolve(a, b, mode='constant', cval=0.0))