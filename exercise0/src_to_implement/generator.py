import os.path
import json
import scipy.misc
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        global fpath, lpath, bsize, imsize, rotate, mirror, shuff, dlabel, index, shArray
        self.fpath = file_path
        self.lpath = label_path
        self.bsize = batch_size
        self.imsize = image_size
        self.rotate = rotation
        self.mirror = mirroring
        self.shuff = shuffle
        f = open(label_path, )                                                               #Opening the file of label path
        self.dlabel = json.loads(f.read())                                                   #Loading lables from the jason file
        shArray = np.random.choice(100, (int(100 / self.bsize), self.bsize), replace=False)  #Generates random shuffle array sample
        self.index = 0
        f.close()                                                                            #Closing the file

    @property
    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        nshape = (np.load('exercise_data/0.npy')).shape
        images = np.empty((self.bsize, 32, 32, 3), dtype=float)                         #creating empty image array
        labels = []                                                                     #Creating Label List
        i = 0
        srow = 0                                                                        #Rows of the shuffle array

        if self.shuff:                                                                  #Shuffle is true case
            srow = shArray[self.index]                                                  #This gives us 1 row of shuffle array
            self.index += 1
            if self.index == int(100 / self.bsize):                                     #End of shuffle array
                self.index = 0                                                          #Moves to the first row

            while i < self.bsize:
                images[i] = np.load('exercise_data/{}.npy'.format(int(srow[i])))        #One row of the shuffe array is stored with image from the exercise_data file
                labels.append(self.dlabel[str(srow[i])])                                #Corresponding labels for the above row of shuffle array is stored here
                i += 1
        else:
            while i < self.bsize:                                                       #Shuffle is not true case
                images[i] = np.load("exercise_data/{}.npy".format(int(self.index)))     #Sequentially loads the image from the exercise_data file
                labels.append(self.dlabel[str(self.index)])                             #Corresponding labels for the above array is stored here
                self.index += 1
                i += 1
                if self.index > len(self.dlabel)-1:                                                     #Once 1 epoch ends start again
                    self.index = 0

        if nshape != self.imsize:                                                  #Checking if the image size matches
            image = np.empty((self.bsize, self.imsize[0], self.imsize[1], 3), dtype=float) #Creating image array for new size of image
            i = 0
            for im in images:                                                           # resizing all the images
                image[i] = resize(im, (self.imsize[0], self.imsize[1]), anti_aliasing=True)
                i += 1
            images = image                                                              #Replacing the old array with resized images

        if self.mirror or self.rotate:                                                  #If the image is rotated or mirrored, call the augment function
            i = 0
            for im in images:
                images[i] = self.augment(im)                                            #Apply augment to each images depending on the condition
                i += 1

        return images, labels                                                           #Return the images and the corresponding labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        if self.rotate:
            m = [1, 2, 3]
            img = np.rot90(img, m[np.random.randint(0, 2)])                             #Randomly choosing the number of time to rotate the image by 90 degrees
        if self.mirror:
            x = np.random.randint(0, 1)                                                 #Randomly choosing whether to flip horizontally or vertically
            if x == 0:
                img = np.flipud(img)                                                    #Flip image array in the up/down direction
            else:
                img = np.fliplr(img)                                                    #Flip image array in the left/right direction
        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        batches, labels = self.next
        fig = plt.figure(figsize=(16, 16))
        rows = int(len(batches) / 2)
        columns = rows
        for i in range(self.bsize):
            img = batches[i]
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(img.astype('uint8'))
            plt.axis('off')
            plt.title(self.class_name(labels[i]))
        plt.show()
